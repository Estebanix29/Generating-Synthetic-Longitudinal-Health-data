# TSTR evaluation with CCS code exclusion

import torch
import pickle
import random
import numpy as np
import yaml
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.001
EPOCHS = 25
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_ccs_label_codes(yaml_path='hcup_ccs_2015_definitions_benchmark.yaml', code_to_index=None):
    """
    Load CCS definitions and return set of code INDICES that define labels.
    These codes will be EXCLUDED from classifier input to prevent leakage.
    """
    with open(yaml_path) as f:
        definitions = yaml.full_load(f)
    
    label_code_indices = set()
    label_code_strings = set()
    
    for group in definitions:
        if not definitions[group].get('use_in_benchmark', False):
            continue
        codes = definitions[group].get('codes', [])
        for code in codes:
            label_code_strings.add(code)
            if code_to_index and code in code_to_index:
                label_code_indices.add(code_to_index[code])
    
    print(f"Loaded {len(label_code_strings)} label-defining codes")
    print(f"Found {len(label_code_indices)} in vocabulary (will be excluded from input)")
    
    return label_code_indices


class Config:
    def __init__(self):
        try:
            import sys
            sys.path.insert(0, '.')
            from config import HALOConfig
            halo_config = HALOConfig()
            self.code_vocab_size = halo_config.code_vocab_size
            self.n_ctx = halo_config.n_ctx
        except:
            self.code_vocab_size = 6984
            self.n_ctx = 48


config = Config()
print(f"Config: code_vocab_size={config.code_vocab_size}, n_ctx={config.n_ctx}")


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


def get_batch_no_leakage(ehr_dataset, loc, batch_size, label_idx, vocab_size, max_ctx, excluded_codes):
    """
    Get batch with label-defining codes EXCLUDED from input.
    
    This forces the classifier to predict from correlated codes only,
    making the task non-trivial (like the paper's evaluation).
    """
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), max_ctx, vocab_size))
    batch_labels = np.array([p['labels'][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr))
    
    for i, p in enumerate(ehr):
        visits = p['visits']
        n_visits = min(len(visits), max_ctx)
        batch_lens[i] = max(n_visits, 1)  # At least 1 to avoid errors
        
        for j in range(n_visits):
            v = visits[j]
            for code in v:
                # EXCLUDE label-defining codes from input
                if code < vocab_size and code not in excluded_codes:
                    batch_ehr[i, j, code] = 1

    return batch_ehr, batch_labels, batch_lens


def train_model(model, train_dataset, val_dataset, vocab_size, max_ctx, label_idx, excluded_codes):
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
                train_dataset, i, BATCH_SIZE, label_idx, vocab_size, max_ctx, excluded_codes)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch_no_leakage(
                    val_dataset, v_i, BATCH_SIZE, label_idx, vocab_size, max_ctx, excluded_codes)
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


def test_model(model, test_dataset, vocab_size, max_ctx, label_idx, excluded_codes):
    """Test a diagnosis model and return metrics."""
    pred_list = []
    true_list = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch_ehr, batch_labels, batch_lens = get_batch_no_leakage(
                test_dataset, i, BATCH_SIZE, label_idx, vocab_size, max_ctx, excluded_codes)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())
    
    round_list = np.around(pred_list)
    
    if len(set(true_list)) < 2:
        return {'auroc': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    try:
        auroc = metrics.roc_auc_score(true_list, pred_list)
    except:
        auroc = 0.5
    
    try:
        f1 = metrics.f1_score(true_list, round_list, zero_division=0)
        precision = metrics.precision_score(true_list, round_list, zero_division=0)
        recall = metrics.recall_score(true_list, round_list, zero_division=0)
    except:
        f1 = 0.0
        precision = 0.0
        recall = 0.0
    
    return {
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def load_datasets(data_dir='./data', synth_path='./results/datasets/haloDataset.pkl'):
    """Load real and synthetic datasets."""
    print("Loading datasets...")
    
    train_dataset = pickle.load(open(os.path.join(data_dir, 'trainDataset.pkl'), 'rb'))
    val_dataset = pickle.load(open(os.path.join(data_dir, 'valDataset.pkl'), 'rb'))
    test_dataset = pickle.load(open(os.path.join(data_dir, 'testDataset.pkl'), 'rb'))
    
    print(f"Real data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    try:
        synth_dataset = pickle.load(open(synth_path, 'rb'))
        synth_dataset = [p for p in synth_dataset if len(p.get('visits', [])) > 0]
        print(f"Synthetic data: {len(synth_dataset)} patients")
    except Exception as e:
        print(f"Warning: Could not load synthetic data from {synth_path}: {e}")
        synth_dataset = None
    
    try:
        id_to_label = pickle.load(open(os.path.join(data_dir, 'idToLabel.pkl'), 'rb'))
    except:
        id_to_label = [f"Label_{i}" for i in range(25)]
    
    # Load code_to_index for mapping CCS codes
    try:
        code_to_index = pickle.load(open(os.path.join(data_dir, 'codeToIndex.pkl'), 'rb'))
    except:
        code_to_index = None
    
    return train_dataset, val_dataset, test_dataset, synth_dataset, id_to_label, code_to_index


def run_evaluation():
    """Run the full TSTR evaluation without label leakage."""
    # Load data
    train_data, val_data, test_data, synth_data, id_to_label, code_to_index = load_datasets()
    
    if synth_data is None:
        print("Cannot proceed without synthetic data!")
        return
    
    # Load label-defining codes to exclude
    excluded_codes = load_ccs_label_codes(code_to_index=code_to_index)
    
    results = {
        'methodology': 'TSTR WITHOUT label leakage (label-defining codes excluded from input)',
        'seed': SEED,
        'num_train': NUM_TRAIN_EXAMPLES,
        'num_test': NUM_TEST_EXAMPLES,
        'vocab_size': config.code_vocab_size,
        'excluded_codes': len(excluded_codes),
        'per_label_results': []
    }
    
    real_aurocs = []
    tstr_aurocs = []
    
    for label_idx in LABEL_IDX_LIST:
        label_name = id_to_label[label_idx] if isinstance(id_to_label, list) and label_idx < len(id_to_label) else f"Label_{label_idx}"
        print(f"\n=== Processing Label {label_idx}: {label_name} ===")
        
        # Prepare balanced datasets
        train_pos = [p for p in train_data if p['labels'][label_idx] == 1]
        train_neg = [p for p in train_data if p['labels'][label_idx] == 0]
        val_pos = [p for p in val_data if p['labels'][label_idx] == 1]
        val_neg = [p for p in val_data if p['labels'][label_idx] == 0]
        test_pos = [p for p in test_data if p['labels'][label_idx] == 1]
        test_neg = [p for p in test_data if p['labels'][label_idx] == 0]
        
        synth_pos = [p for p in synth_data if p['labels'][label_idx] == 1]
        synth_neg = [p for p in synth_data if p['labels'][label_idx] == 0]
        
        print(f"  Real: {len(train_pos)} pos, {len(train_neg)} neg")
        print(f"  Synthetic: {len(synth_pos)} pos, {len(synth_neg)} neg")
        
        if len(train_pos) < 10 or len(synth_pos) < 10:
            print(f"  Skipping - not enough positive samples")
            continue
        
        # Sample balanced datasets
        n_train_half = min(NUM_TRAIN_EXAMPLES // 2, len(train_pos), len(train_neg))
        n_test_half = min(NUM_TEST_EXAMPLES // 2, len(test_pos), len(test_neg))
        n_val_half = min(NUM_VAL_EXAMPLES // 2, len(val_pos), len(val_neg))
        n_synth_half = min(NUM_TRAIN_EXAMPLES // 2, len(synth_pos), len(synth_neg))
        
        train_real = list(np.random.choice(train_pos, n_train_half, replace=len(train_pos)<n_train_half)) + \
                     list(np.random.choice(train_neg, n_train_half, replace=len(train_neg)<n_train_half))
        train_synth = list(np.random.choice(synth_pos, n_synth_half, replace=len(synth_pos)<n_synth_half)) + \
                      list(np.random.choice(synth_neg, n_synth_half, replace=len(synth_neg)<n_synth_half))
        val_set = list(np.random.choice(val_pos, n_val_half, replace=len(val_pos)<n_val_half)) + \
                  list(np.random.choice(val_neg, n_val_half, replace=len(val_neg)<n_val_half))
        test_set = list(np.random.choice(test_pos, n_test_half, replace=len(test_pos)<n_test_half)) + \
                   list(np.random.choice(test_neg, n_test_half, replace=len(test_neg)<n_test_half))
        
        # Train on real data (without label codes in input)
        print(f"  Training on real data (excluding {len(excluded_codes)} label codes)...")
        model_real = DiagnosisModel(config.code_vocab_size).to(device)
        model_real = train_model(model_real, train_real, val_set, config.code_vocab_size, config.n_ctx, label_idx, excluded_codes)
        real_metrics = test_model(model_real, test_set, config.code_vocab_size, config.n_ctx, label_idx, excluded_codes)
        
        # Train on synthetic data (without label codes in input)
        print(f"  Training on synthetic data...")
        model_synth = DiagnosisModel(config.code_vocab_size).to(device)
        model_synth = train_model(model_synth, train_synth, val_set, config.code_vocab_size, config.n_ctx, label_idx, excluded_codes)
        tstr_metrics = test_model(model_synth, test_set, config.code_vocab_size, config.n_ctx, label_idx, excluded_codes)
        
        # Record results
        utility_pct = (tstr_metrics['auroc'] / real_metrics['auroc'] * 100) if real_metrics['auroc'] > 0 else 0
        
        label_result = {
            'label_idx': label_idx,
            'label_name': label_name,
            'real_auroc': real_metrics['auroc'],
            'tstr_auroc': tstr_metrics['auroc'],
            'utility_pct': utility_pct,
            'real_metrics': real_metrics,
            'tstr_metrics': tstr_metrics
        }
        results['per_label_results'].append(label_result)
        
        real_aurocs.append(real_metrics['auroc'])
        tstr_aurocs.append(tstr_metrics['auroc'])
        
        print(f"  Real AUROC: {real_metrics['auroc']:.4f}")
        print(f"  TSTR AUROC: {tstr_metrics['auroc']:.4f}")
        print(f"  Utility: {utility_pct:.1f}%")
    
    # Summary
    results['summary'] = {
        'num_labels': len(real_aurocs),
        'avg_real_auroc': np.mean(real_aurocs) if real_aurocs else 0,
        'avg_tstr_auroc': np.mean(tstr_aurocs) if tstr_aurocs else 0,
        'avg_utility_pct': np.mean([r['utility_pct'] for r in results['per_label_results']]) if results['per_label_results'] else 0
    }
    
    print("\n" + "-"*60)
    print("Summary (without label leakage)")
    print("-"*60)
    print(f"Labels evaluated: {results['summary']['num_labels']}")
    print(f"Excluded codes: {len(excluded_codes)}")
    print(f"Average Real AUROC: {results['summary']['avg_real_auroc']:.4f}")
    print(f"Average TSTR AUROC: {results['summary']['avg_tstr_auroc']:.4f}")
    print(f"Average Utility: {results['summary']['avg_utility_pct']:.1f}%")
    print()
    print("Expected: Real AUROC should now be ~0.94 (like paper), not 0.99")
    
    # Save results
    output_path = './tstr_ccs_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_evaluation()
