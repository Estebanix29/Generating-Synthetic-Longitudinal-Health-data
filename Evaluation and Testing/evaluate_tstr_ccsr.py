"""
TSTR Evaluation with CCSR (ICD-10 Native) Labels
=================================================

This script uses CCSR (Clinical Classifications Software Refined) which is
designed for ICD-10 codes, unlike CCS which was designed for ICD-9.

CCSR provides better coverage for ICD-10 codes and should give
more meaningful evaluation results.

Key differences from CCS-based evaluation:
1. Uses ICD-10 native phenotype definitions
2. Better coverage of ICD-10 vocabulary
3. More clinically meaningful label categories
"""

import torch
import pickle
import random
import numpy as np
import re
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Configuration
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.001
EPOCHS = 25
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500

# MIMIC-IV specific paths
DATA_DIR = './data'
SYNTH_PATH = './data/haloDataset_original.pkl'  # Updated path

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CCSR definitions for ICD-10 codes (25 most common diagnostic categories)
CCSR_DEFINITIONS = {
    # Cardiovascular
    "Heart failure": {
        "codes": ["I50", "I110", "I130", "I132"],  # Heart failure codes
        "use_in_benchmark": True
    },
    "Acute myocardial infarction": {
        "codes": ["I21", "I22"],  # AMI codes
        "use_in_benchmark": True
    },
    "Cardiac dysrhythmias": {
        "codes": ["I47", "I48", "I49", "R001"],  # Arrhythmia codes
        "use_in_benchmark": True
    },
    "Coronary atherosclerosis": {
        "codes": ["I25"],  # Coronary artery disease
        "use_in_benchmark": True
    },
    "Essential hypertension": {
        "codes": ["I10"],  # Primary hypertension
        "use_in_benchmark": True
    },
    "Hypertension with complications": {
        "codes": ["I11", "I12", "I13", "I15"],  # Secondary/complicated HTN
        "use_in_benchmark": True
    },
    
    # Respiratory
    "Pneumonia": {
        "codes": ["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
        "use_in_benchmark": True
    },
    "Respiratory failure": {
        "codes": ["J96", "J80"],  # Respiratory failure, ARDS
        "use_in_benchmark": True
    },
    "COPD and bronchiectasis": {
        "codes": ["J40", "J41", "J42", "J43", "J44", "J47"],
        "use_in_benchmark": True
    },
    "Asthma": {
        "codes": ["J45", "J46"],
        "use_in_benchmark": True
    },
    
    # Renal
    "Acute kidney failure": {
        "codes": ["N17"],  # Acute kidney injury
        "use_in_benchmark": True
    },
    "Chronic kidney disease": {
        "codes": ["N18", "N19"],  # CKD
        "use_in_benchmark": True
    },
    
    # Metabolic/Endocrine
    "Diabetes mellitus": {
        "codes": ["E08", "E09", "E10", "E11", "E13"],  # All diabetes types
        "use_in_benchmark": True
    },
    "Disorders of lipid metabolism": {
        "codes": ["E78"],  # Hyperlipidemia
        "use_in_benchmark": True
    },
    "Fluid and electrolyte disorders": {
        "codes": ["E86", "E87"],
        "use_in_benchmark": True
    },
    
    # Infectious
    "Septicemia": {
        "codes": ["A40", "A41", "R652"],  # Sepsis codes
        "use_in_benchmark": True
    },
    "Urinary tract infections": {
        "codes": ["N39"],  # UTI
        "use_in_benchmark": True
    },
    
    # Gastrointestinal
    "Gastrointestinal hemorrhage": {
        "codes": ["K25", "K26", "K27", "K28", "K29", "K92"],  # GI bleeding
        "use_in_benchmark": True
    },
    "Liver disease": {
        "codes": ["K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77"],
        "use_in_benchmark": True
    },
    
    # Neurological
    "Cerebrovascular disease": {
        "codes": ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
        "use_in_benchmark": True
    },
    
    # Other
    "Shock": {
        "codes": ["R57"],  # All shock types
        "use_in_benchmark": True
    },
    "Anemia": {
        "codes": ["D50", "D51", "D52", "D53", "D64"],
        "use_in_benchmark": True
    },
    "Coagulation and hemorrhagic disorders": {
        "codes": ["D65", "D66", "D67", "D68", "D69"],
        "use_in_benchmark": True
    },
    "Mental disorders": {
        "codes": ["F01", "F02", "F03", "F04", "F05", "F06", "F07"],  # Organic mental disorders
        "use_in_benchmark": True
    },
    "Complications of care": {
        "codes": ["T80", "T81", "T82", "T83", "T84", "T85", "T86", "T87", "T88"],
        "use_in_benchmark": True
    },
}


def normalize_icd10_code(code):
    """
    Normalize ICD-10 code for matching.
    Removes dots and converts to uppercase.
    """
    if code is None:
        return ""
    return str(code).upper().replace(".", "").strip()


def code_matches_prefix(code, prefix):
    """
    Check if an ICD-10 code matches a CCSR prefix.
    E.g., code "I5021" matches prefix "I50"
    """
    normalized_code = normalize_icd10_code(code)
    normalized_prefix = normalize_icd10_code(prefix)
    return normalized_code.startswith(normalized_prefix)


def build_ccsr_mappings(index_to_code):
    """
    Build CCSR label mappings from the vocabulary.
    
    Returns:
        code_to_group: dict mapping code string -> label name
        label_names: list of label names
        label_code_indices: set of code indices that define labels
    """
    code_to_group = {}
    label_names = []
    label_code_indices = set()
    
    # Get all labels that are marked for benchmark
    for label_name, label_info in CCSR_DEFINITIONS.items():
        if label_info.get('use_in_benchmark', False):
            label_names.append(label_name)
    
    label_names = sorted(label_names)
    
    # Build reverse mapping: code_index -> code_string
    # index_to_code should be provided
    
    # Map codes to labels
    for code_idx, code_str in index_to_code.items():
        normalized = normalize_icd10_code(code_str)
        
        for label_name, label_info in CCSR_DEFINITIONS.items():
            if not label_info.get('use_in_benchmark', False):
                continue
            
            for prefix in label_info['codes']:
                if code_matches_prefix(code_str, prefix):
                    code_to_group[code_str] = label_name
                    label_code_indices.add(code_idx)
                    break  # Code matched this label, move to next code
    
    print(f"CCSR Mapping Statistics:")
    print(f"  Total labels: {len(label_names)}")
    print(f"  Codes mapped to labels: {len(code_to_group)}")
    print(f"  Code indices for exclusion: {len(label_code_indices)}")
    
    # Show coverage per label
    label_coverage = {}
    for label_name in label_names:
        count = sum(1 for c, g in code_to_group.items() if g == label_name)
        label_coverage[label_name] = count
    
    print(f"\nPer-label code coverage:")
    for label, count in sorted(label_coverage.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} codes")
    
    return code_to_group, label_names, label_code_indices


def assign_ccsr_labels(data, code_to_group, label_names, index_to_code):
    """
    Assign CCSR-based labels to patients.
    
    Args:
        data: list of patient dicts with 'visits' (list of list of code indices)
        code_to_group: dict mapping code string -> label name
        label_names: sorted list of label names
        index_to_code: dict mapping code index -> code string
    
    Returns:
        data with 'ccsr_labels' added to each patient
    """
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)
    
    label_counts = np.zeros(num_labels)
    
    for patient in data:
        labels = np.zeros(num_labels)
        
        for visit in patient['visits']:
            for code_idx in visit:
                if code_idx in index_to_code:
                    code_str = index_to_code[code_idx]
                    if code_str in code_to_group:
                        label_name = code_to_group[code_str]
                        if label_name in label_to_idx:
                            labels[label_to_idx[label_name]] = 1
        
        patient['ccsr_labels'] = labels
        label_counts += labels
    
    print(f"\nLabel distribution in dataset:")
    for i, name in enumerate(label_names):
        print(f"  {name}: {int(label_counts[i])} patients ({100*label_counts[i]/len(data):.1f}%)")
    
    return data


class Config:
    """MIMIC-IV configuration."""
    def __init__(self):
        try:
            import sys
            sys.path.insert(0, '.')
            from config import HALOConfig
            halo_config = HALOConfig()
            self.code_vocab_size = halo_config.code_vocab_size
            self.n_ctx = halo_config.n_ctx
            print(f"Loaded config from HALOConfig")
        except Exception as e:
            self.code_vocab_size = 28562
            self.n_ctx = 240
            print(f"Using MIMIC-IV default config")


class DiagnosisModel(nn.Module):
    """LSTM classifier for diagnosis prediction."""
    def __init__(self, vocab_size):
        super(DiagnosisModel, self).__init__()
        self.embedding = nn.Linear(vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)
        
        return prob


def get_batch_no_leakage(ehr_dataset, loc, batch_size, label_idx, vocab_size, max_ctx, excluded_codes, use_ccsr=True):
    """
    Get batch with label-defining codes EXCLUDED from input.
    """
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), max_ctx, vocab_size))
    
    label_key = 'ccsr_labels' if use_ccsr else 'labels'
    batch_labels = np.array([p[label_key][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr))
    
    for i, p in enumerate(ehr):
        visits = p['visits']
        n_visits = min(len(visits), max_ctx)
        batch_lens[i] = max(n_visits, 1)
        
        for j in range(n_visits):
            v = visits[j]
            for code in v:
                if code < vocab_size and code not in excluded_codes:
                    batch_ehr[i, j, code] = 1

    return batch_ehr, batch_labels, batch_lens


def train_model(model, train_dataset, val_dataset, vocab_size, max_ctx, label_idx, excluded_codes, use_ccsr=True):
    """Train a diagnosis model."""
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    best_state = None
    
    for e in range(EPOCHS):
        np.random.shuffle(train_dataset)
        train_losses = []
        
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels, batch_lens = get_batch_no_leakage(
                train_dataset, i, BATCH_SIZE, label_idx, vocab_size, max_ctx, excluded_codes, use_ccsr)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch_no_leakage(
                    val_dataset, v_i, BATCH_SIZE, label_idx, vocab_size, max_ctx, excluded_codes, use_ccsr)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            
            cur_val_loss = np.mean(val_losses)
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                best_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
    
    if best_state:
        model.load_state_dict(best_state['model'])
    
    return model


def test_model(model, test_dataset, vocab_size, max_ctx, label_idx, excluded_codes, use_ccsr=True):
    """Test a diagnosis model and return metrics."""
    pred_list = []
    true_list = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch_ehr, batch_labels, batch_lens = get_batch_no_leakage(
                test_dataset, i, BATCH_SIZE, label_idx, vocab_size, max_ctx, excluded_codes, use_ccsr)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())
    
    if len(set(true_list)) < 2:
        return {'auroc': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    try:
        auroc = metrics.roc_auc_score(true_list, pred_list)
    except:
        auroc = 0.5
    
    round_list = np.around(pred_list)
    try:
        f1 = metrics.f1_score(true_list, round_list, zero_division=0)
        precision = metrics.precision_score(true_list, round_list, zero_division=0)
        recall = metrics.recall_score(true_list, round_list, zero_division=0)
    except:
        f1, precision, recall = 0.0, 0.0, 0.0
    
    return {
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    print("=" * 70)
    print("CCSR-Based TSTR Evaluation (ICD-10 Native) - MIMIC-IV")
    print("=" * 70)
    
    config = Config()
    print(f"Config: code_vocab_size={config.code_vocab_size}, n_ctx={config.n_ctx}")
    
    # Load vocabulary mappings
    print("\nLoading vocabulary mappings...")
    try:
        # Try different paths
        possible_paths = [
            os.path.join(DATA_DIR, 'codeToIndex.pkl'),
            os.path.join(DATA_DIR, 'code_to_index.pkl'),
            './codeToIndex.pkl',
            './data/codeToIndex.pkl',
        ]
        
        code_to_index = None
        for path in possible_paths:
            if os.path.exists(path):
                code_to_index = pickle.load(open(path, 'rb'))
                print(f"Loaded code_to_index from {path}")
                break
        
        if code_to_index is None:
            raise FileNotFoundError("Could not find codeToIndex.pkl")
        
        index_to_code = {v: k for k, v in code_to_index.items()}
        print(f"Vocabulary size: {len(code_to_index)}")
        
        # Show sample codes to verify ICD-10 format
        sample_codes = list(code_to_index.keys())[:20]
        print(f"Sample codes: {sample_codes}")
        
        # Count ICD-9 vs ICD-10 codes
        icd10_count = sum(1 for c in code_to_index.keys() if isinstance(c, str) and len(c) > 0 and c[0].isalpha())
        icd9_count = len(code_to_index) - icd10_count
        print(f"ICD-10 codes (alphanumeric start): {icd10_count}")
        print(f"ICD-9 codes (numeric start): {icd9_count}")
        
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        print("Creating dummy vocabulary for testing...")
        code_to_index = {}
        index_to_code = {}
    
    # Build CCSR mappings
    print("\nBuilding CCSR label mappings...")
    code_to_group, label_names, excluded_codes = build_ccsr_mappings(index_to_code)
    
    num_labels = len(label_names)
    print(f"\nTotal CCSR labels for evaluation: {num_labels}")
    print(f"Codes to exclude from classifier input: {len(excluded_codes)}")
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        # Try different file naming conventions
        train_paths = [
            os.path.join(DATA_DIR, 'trainDataset.pkl'),
            os.path.join(DATA_DIR, 'dataset_train.pkl'),
            './trainDataset.pkl',
        ]
        test_paths = [
            os.path.join(DATA_DIR, 'testDataset.pkl'),
            os.path.join(DATA_DIR, 'dataset_test.pkl'),
            './testDataset.pkl',
        ]
        val_paths = [
            os.path.join(DATA_DIR, 'valDataset.pkl'),
            os.path.join(DATA_DIR, 'dataset_val.pkl'),
            './valDataset.pkl',
        ]
        
        train_dataset = None
        for path in train_paths:
            if os.path.exists(path):
                train_dataset = pickle.load(open(path, 'rb'))
                print(f"Loaded train from {path}")
                break
        
        test_dataset = None
        for path in test_paths:
            if os.path.exists(path):
                test_dataset = pickle.load(open(path, 'rb'))
                print(f"Loaded test from {path}")
                break
        
        val_dataset = None
        for path in val_paths:
            if os.path.exists(path):
                val_dataset = pickle.load(open(path, 'rb'))
                print(f"Loaded val from {path}")
                break
        
        if train_dataset is None or test_dataset is None:
            raise FileNotFoundError("Could not find train/test datasets")
        
        if val_dataset is None:
            # Create validation from training
            from sklearn.model_selection import train_test_split
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=4)
            print("Created validation split from training data")
        
        print(f"Real data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Assign CCSR labels
    print("\nAssigning CCSR labels to real datasets...")
    train_dataset = assign_ccsr_labels(train_dataset, code_to_group, label_names, index_to_code)
    val_dataset = assign_ccsr_labels(val_dataset, code_to_group, label_names, index_to_code)
    test_dataset = assign_ccsr_labels(test_dataset, code_to_group, label_names, index_to_code)
    
    # Load synthetic data
    print("\nLoading synthetic data...")
    try:
        # Try multiple paths
        synth_paths = [
            SYNTH_PATH,
            './data/haloDataset_original.pkl',
            './data/haloDataset.pkl',
            './results/datasets/haloDataset.pkl',
        ]
        
        synth_dataset = None
        for spath in synth_paths:
            if os.path.exists(spath):
                synth_dataset = pickle.load(open(spath, 'rb'))
                print(f"Loaded synthetic data from {spath}")
                break
        
        if synth_dataset is None:
            raise FileNotFoundError("Could not find synthetic data file")
            
        synth_dataset = [p for p in synth_dataset if len(p.get('visits', [])) > 0]
        print(f"Synthetic data: {len(synth_dataset)} patients")
        
        # Assign CCSR labels to synthetic data
        print("Assigning CCSR labels to synthetic dataset...")
        synth_dataset = assign_ccsr_labels(synth_dataset, code_to_group, label_names, index_to_code)
    except Exception as e:
        print(f"Error loading synthetic data: {e}")
        synth_dataset = None
    
    # Sample datasets for faster evaluation
    print("\nSampling datasets...")
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    random.shuffle(val_dataset)
    
    train_sample = train_dataset[:NUM_TRAIN_EXAMPLES]
    val_sample = val_dataset[:NUM_VAL_EXAMPLES]
    test_sample = test_dataset[:NUM_TEST_EXAMPLES]
    
    if synth_dataset:
        random.shuffle(synth_dataset)
        synth_sample = synth_dataset[:NUM_TRAIN_EXAMPLES]
    else:
        synth_sample = None
    
    print(f"Sample sizes: train={len(train_sample)}, val={len(val_sample)}, test={len(test_sample)}")
    
    # Run evaluation
    print("\n" + "=" * 70)
    print("RUNNING CCSR-BASED NO-LEAKAGE EVALUATION")
    print("=" * 70)
    
    results = {
        'labels': [],
        'real_auroc': [],
        'tstr_auroc': [],
        'utility': []
    }
    
    for label_idx, label_name in enumerate(label_names):
        print(f"\n--- Label {label_idx}: {label_name} ---")
        
        # Check label distribution
        train_pos = sum(p['ccsr_labels'][label_idx] for p in train_sample)
        test_pos = sum(p['ccsr_labels'][label_idx] for p in test_sample)
        
        print(f"Train positives: {train_pos}/{len(train_sample)} ({100*train_pos/len(train_sample):.1f}%)")
        print(f"Test positives: {test_pos}/{len(test_sample)} ({100*test_pos/len(test_sample):.1f}%)")
        
        # Skip labels with too few positives
        if train_pos < 10 or test_pos < 5:
            print(f"Skipping label {label_name} - insufficient positive samples")
            continue
        
        # Train on REAL data
        print("Training on REAL data (no leakage)...")
        real_model = DiagnosisModel(config.code_vocab_size).to(device)
        real_model = train_model(real_model, train_sample, val_sample, 
                                  config.code_vocab_size, config.n_ctx, 
                                  label_idx, excluded_codes, use_ccsr=True)
        real_metrics = test_model(real_model, test_sample, 
                                   config.code_vocab_size, config.n_ctx, 
                                   label_idx, excluded_codes, use_ccsr=True)
        print(f"Real AUROC: {real_metrics['auroc']:.4f}")
        
        # Train on SYNTHETIC data
        tstr_auroc = 0.5
        if synth_sample:
            synth_pos = sum(p['ccsr_labels'][label_idx] for p in synth_sample)
            print(f"Synth positives: {synth_pos}/{len(synth_sample)} ({100*synth_pos/len(synth_sample):.1f}%)")
            
            if synth_pos >= 10:
                print("Training on SYNTHETIC data (no leakage)...")
                synth_model = DiagnosisModel(config.code_vocab_size).to(device)
                synth_model = train_model(synth_model, synth_sample, val_sample,
                                          config.code_vocab_size, config.n_ctx,
                                          label_idx, excluded_codes, use_ccsr=True)
                synth_metrics = test_model(synth_model, test_sample,
                                           config.code_vocab_size, config.n_ctx,
                                           label_idx, excluded_codes, use_ccsr=True)
                tstr_auroc = synth_metrics['auroc']
                print(f"TSTR AUROC: {tstr_auroc:.4f}")
            else:
                print("Skipping synthetic training - insufficient positives")
        
        # Calculate utility
        utility = tstr_auroc / real_metrics['auroc'] if real_metrics['auroc'] > 0 else 0
        print(f"Utility: {utility*100:.1f}%")
        
        results['labels'].append(label_name)
        results['real_auroc'].append(real_metrics['auroc'])
        results['tstr_auroc'].append(tstr_auroc)
        results['utility'].append(utility)
    
    # Summary
    print("\n" + "=" * 70)
    print("CCSR-BASED EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\nLabels evaluated: {len(results['labels'])}")
    print(f"Excluded ICD-10 codes: {len(excluded_codes)}")
    
    if results['labels']:
        avg_real = np.mean(results['real_auroc'])
        avg_tstr = np.mean(results['tstr_auroc'])
        avg_utility = np.mean(results['utility'])
        
        print(f"\nAverage Real AUROC: {avg_real:.4f}")
        print(f"Average TSTR AUROC: {avg_tstr:.4f}")
        print(f"Average Utility: {avg_utility*100:.1f}%")
        
        print("\n--- Per-Label Results ---")
        for i, label in enumerate(results['labels']):
            print(f"{label}: Real={results['real_auroc'][i]:.4f}, "
                  f"TSTR={results['tstr_auroc'][i]:.4f}, "
                  f"Utility={results['utility'][i]*100:.1f}%")
    
    # Save results
    results_file = 'tstr_ccsr_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'labels': results['labels'],
            'real_auroc': [float(x) for x in results['real_auroc']],
            'tstr_auroc': [float(x) for x in results['tstr_auroc']],
            'utility': [float(x) for x in results['utility']],
            'avg_real_auroc': float(np.mean(results['real_auroc'])) if results['real_auroc'] else 0,
            'avg_tstr_auroc': float(np.mean(results['tstr_auroc'])) if results['tstr_auroc'] else 0,
            'avg_utility': float(np.mean(results['utility'])) if results['utility'] else 0,
            'excluded_codes_count': len(excluded_codes),
            'num_labels': len(results['labels']),
            'method': 'CCSR (ICD-10 native)'
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
