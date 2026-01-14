import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# MIMIC-IV dataset paths
mimic_dir = "./mimic-iv-3.1/hosp/"
admissionFile = mimic_dir + "admissions.csv.gz"
diagnosisFile = mimic_dir + "diagnoses_icd.csv.gz"

print("Loading CSVs Into Dataframes")
admissionDf = pd.read_csv(admissionFile, dtype={'subject_id': str, 'hadm_id': str})
admissionDf['admittime'] = pd.to_datetime(admissionDf['admittime'])
admissionDf = admissionDf.sort_values('admittime')
admissionDf = admissionDf.reset_index(drop=True)

diagnosisDf = pd.read_csv(diagnosisFile, dtype={'hadm_id': str, 'icd_code': str})
diagnosisDf = diagnosisDf[diagnosisDf['icd_code'].notnull()]

# MIMIC-IV contains both ICD9 and ICD10 codes
# The CCS definitions file uses ICD9, so filtering or conversion may be needed
# For now, use only ICD9 codes to match the original implementation
print(f"Total diagnosis records: {len(diagnosisDf)}")
print(f"ICD9 records: {len(diagnosisDf[diagnosisDf['icd_version'] == 9])}")
print(f"ICD10 records: {len(diagnosisDf[diagnosisDf['icd_version'] == 10])}")

# Option 1: Use only ICD9 codes (simpler, but loses data)
# Uncomment this line to use only ICD9:
# diagnosisDf = diagnosisDf[diagnosisDf['icd_version'] == 9]

# Option 2: Use both ICD9 and ICD10 (more data, but may not map perfectly to CCS)
# Keep both and handle the mapping later
print("\nNote: Using both ICD9 and ICD10 codes. ICD10 codes that don't map to CCS groups will not contribute to labels.")

print("\nBuilding Dataset - Optimized approach")
# Group diagnoses by hadm_id first (much faster than repeated lookups)
diagnosis_dict = diagnosisDf.groupby('hadm_id')['icd_code'].apply(lambda x: list(set(x))).to_dict()

data = {}
for row in tqdm(admissionDf.itertuples(), total=admissionDf.shape[0]):          
    # Extracting Admissions Table Info
    hadm_id = row.hadm_id
    subject_id = row.subject_id
            
    # Extracting the Diagnoses
    diagnoses = diagnosis_dict.get(hadm_id, [])
    
    # Building the hospital admission data point
    if subject_id not in data:
        data[subject_id] = {'visits': [diagnoses]}
    else:
        data[subject_id]['visits'].append(diagnoses)

code_to_index = {}
all_codes = list(set([c for p in data.values() for v in p['visits'] for c in v]))
np.random.seed(4)  # For reproducibility
np.random.shuffle(all_codes)
for c in all_codes:
    code_to_index[c] = len(code_to_index)
print(f"VOCAB SIZE: {len(code_to_index)}")
index_to_code = {v: k for k, v in code_to_index.items()}

data = list(data.values())

print("\nAdding Labels")
with open("hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.full_load(definitions_file)

code_to_group = {}
for group in definitions:
    if definitions[group]['use_in_benchmark'] == False:
        continue
    codes = definitions[group]['codes']
    for code in codes:
        if code not in code_to_group:
            code_to_group[code] = group
        else:
            assert code_to_group[code] == group

id_to_group = sorted([k for k in definitions.keys() if definitions[k]['use_in_benchmark'] == True])
group_to_id = dict((x, i) for (i, x) in enumerate(id_to_group))

# Add Labels
codes_mapped = 0
codes_not_mapped = 0
for p in data:
    label = np.zeros(len(group_to_id))
    for v in p['visits']:
        for c in v:
            if c in code_to_group:
                label[group_to_id[code_to_group[c]]] = 1
                codes_mapped += 1
            else:
                codes_not_mapped += 1
    
    p['labels'] = label

print(f"Codes mapped to CCS groups: {codes_mapped}")
print(f"Codes not mapped to CCS groups: {codes_not_mapped}")
print(f"Mapping coverage: {codes_mapped/(codes_mapped+codes_not_mapped)*100:.2f}%")

print("\nConverting Visits")
for p in data:
    new_visits = []
    for v in p['visits']:
        new_visit = []
        for c in v:
            new_visit.append(code_to_index[c])
                
        new_visits.append((list(set(new_visit))))
        
    p['visits'] = new_visits    

print(f"MAX LEN: {max([len(p['visits']) for p in data])}")
print(f"AVG LEN: {np.mean([len(p['visits']) for p in data]):.2f}")
print(f"MAX VISIT LEN: {max([len(v) for p in data for v in p['visits']])}")
print(f"AVG VISIT LEN: {np.mean([len(v) for p in data for v in p['visits']]):.2f}")
print(f"NUM RECORDS: {len(data)}")
print(f"NUM LONGITUDINAL RECORDS: {len([p for p in data if len(p['visits']) > 1])}")

# Train-Val-Test Split
print("\nSplitting Datasets")
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=4, shuffle=True)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4, shuffle=True)

# Create data directory if it doesn't exist
import os
os.makedirs("./data", exist_ok=True)

# Save Everything
print("\nSaving Everything")
print(f"Code vocabulary size: {len(index_to_code)}")
print(f"Label vocabulary size: {len(id_to_group)}")
print(f"Training records: {len(train_dataset)}")
print(f"Validation records: {len(val_dataset)}")
print(f"Test records: {len(test_dataset)}")

pickle.dump(code_to_index, open("./data/codeToIndex.pkl", "wb"))
pickle.dump(index_to_code, open("./data/indexToCode.pkl", "wb"))
pickle.dump(id_to_group, open("./data/idToLabel.pkl", "wb"))
pickle.dump(train_dataset, open("./data/trainDataset.pkl", "wb"))
pickle.dump(val_dataset, open("./data/valDataset.pkl", "wb"))
pickle.dump(test_dataset, open("./data/testDataset.pkl", "wb"))

print("\nPreprocessing complete. All files saved to ./data/")
print("\nNext steps:")
print("1. Check the vocabulary size and update config.py if needed")
print("2. Create ./save/ directory for model checkpoints")
print("3. Run train_model.py to train the model")
