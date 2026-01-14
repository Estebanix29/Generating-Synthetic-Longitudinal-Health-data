"""
Fidelity Evaluation for HALO Synthetic Data
============================================

This script compares synthetic EHR data to real data across multiple dimensions:

1. Code Frequency Distribution
2. Visit Length Distribution
3. Patient Length Distribution
4. Code Co-occurrence
5. Label Distribution
6. Temporal Patterns 

Metrics:
- Jensen-Shannon Divergence (JSD): 0 = identical, 1 = completely different
- Pearson Correlation: 1 = perfect correlation, 0 = no correlation
- Kolmogorov-Smirnov Test: p-value > 0.05 suggests similar distributions

Works with both MIMIC-III and MIMIC-IV.
"""

import pickle
import numpy as np
import os
import json
from collections import Counter
from scipy import stats
from scipy.spatial.distance import jensenshannon
import argparse

DATA_DIR = './data'
SYNTH_PATH = './results/datasets/haloDataset.pkl'


def load_data(data_dir, synth_path):
    print("Loading datasets...")
    
    train_data = pickle.load(open(os.path.join(data_dir, 'trainDataset.pkl'), 'rb'))
    test_data = pickle.load(open(os.path.join(data_dir, 'testDataset.pkl'), 'rb'))
    real_data = train_data + test_data
    
    synth_data = pickle.load(open(synth_path, 'rb'))
    synth_data = [p for p in synth_data if len(p.get('visits', [])) > 0]
    
    print(f"Real patients: {len(real_data)}")
    print(f"Synthetic patients: {len(synth_data)}")
    
    # Try to load vocabulary info
    try:
        id_to_code = pickle.load(open(os.path.join(data_dir, 'idToCode.pkl'), 'rb'))
        vocab_size = len(id_to_code)
    except:
        # Infer vocab size from data
        all_codes = set()
        for p in real_data + synth_data:
            for v in p['visits']:
                all_codes.update(v)
        vocab_size = max(all_codes) + 1 if all_codes else 10000
        id_to_code = None
    
    print(f"Vocabulary size: {vocab_size}")
    
    return real_data, synth_data, vocab_size, id_to_code


def compute_code_frequencies(data, vocab_size):
    """Compute frequency of each code across all patients."""
    freq = np.zeros(vocab_size)
    total_codes = 0
    
    for patient in data:
        for visit in patient['visits']:
            for code in visit:
                if code < vocab_size:
                    freq[code] += 1
                    total_codes += 1
    
    # Normalize to probabilities
    if total_codes > 0:
        freq = freq / total_codes
    
    return freq


def compute_visit_lengths(data):
    """Compute distribution of visit lengths (codes per visit)."""
    lengths = []
    for patient in data:
        for visit in patient['visits']:
            lengths.append(len(visit))
    return lengths


def compute_patient_lengths(data):
    """Compute distribution of patient lengths (visits per patient)."""
    lengths = []
    for patient in data:
        lengths.append(len(patient['visits']))
    return lengths


def compute_code_cooccurrence(data, vocab_size, top_k=100):
    """Compute co-occurrence matrix for top-k most frequent codes."""
    # First find top-k codes
    code_counts = Counter()
    for patient in data:
        for visit in patient['visits']:
            code_counts.update(visit)
    
    top_codes = [c for c, _ in code_counts.most_common(top_k) if c < vocab_size]
    code_to_idx = {c: i for i, c in enumerate(top_codes)}
    
    # Compute co-occurrence
    cooc = np.zeros((len(top_codes), len(top_codes)))
    for patient in data:
        for visit in patient['visits']:
            visit_codes = [c for c in visit if c in code_to_idx]
            for i, c1 in enumerate(visit_codes):
                for c2 in visit_codes[i:]:
                    idx1, idx2 = code_to_idx[c1], code_to_idx[c2]
                    cooc[idx1, idx2] += 1
                    if idx1 != idx2:
                        cooc[idx2, idx1] += 1
    
    # Normalize
    total = cooc.sum()
    if total > 0:
        cooc = cooc / total
    
    return cooc, top_codes


def compute_label_distribution(data):
    """Compute distribution of labels."""
    if 'labels' not in data[0]:
        return None
    
    num_labels = len(data[0]['labels'])
    label_counts = np.zeros(num_labels)
    
    for patient in data:
        label_counts += patient['labels']
    
    # Normalize
    label_freq = label_counts / len(data)
    return label_freq


def compute_temporal_distribution(data, vocab_size, max_visits=10):
    """Compute how codes are distributed across visit positions."""
    # For each code, compute what fraction appears in visit 1, 2, 3, etc.
    temporal = np.zeros((vocab_size, max_visits))
    
    for patient in data:
        for v_idx, visit in enumerate(patient['visits'][:max_visits]):
            for code in visit:
                if code < vocab_size:
                    temporal[code, v_idx] += 1
    
    # Normalize per code
    row_sums = temporal.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    temporal = temporal / row_sums
    
    return temporal


def jensen_shannon_divergence(p, q):
    """Compute Jensen-Shannon divergence between two distributions."""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.array(p) + eps
    q = np.array(q) + eps
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return jensenshannon(p, q)


def ks_test(dist1, dist2):
    """Perform Kolmogorov-Smirnov test."""
    stat, pvalue = stats.ks_2samp(dist1, dist2)
    return stat, pvalue


def correlation(x, y):
    """Compute Pearson correlation."""
    # Handle edge cases
    if len(x) == 0 or len(y) == 0:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    corr, _ = stats.pearsonr(x.flatten(), y.flatten())
    return corr


def evaluate_fidelity(real_data, synth_data, vocab_size, id_to_code=None):
    """Run all fidelity evaluations."""
    results = {}
    
    print("\n" + "-"*70)
    print("Fidelity evaluation")
    print("-"*70)
    
    # 1. Code Frequency Distribution
    print("\n1. Code Frequency Distribution...")
    real_freq = compute_code_frequencies(real_data, vocab_size)
    synth_freq = compute_code_frequencies(synth_data, vocab_size)
    
    # Only compare codes that appear in either dataset
    nonzero_mask = (real_freq > 0) | (synth_freq > 0)
    real_freq_nz = real_freq[nonzero_mask]
    synth_freq_nz = synth_freq[nonzero_mask]
    
    jsd_code = jensen_shannon_divergence(real_freq_nz, synth_freq_nz)
    corr_code = correlation(real_freq_nz, synth_freq_nz)
    
    print(f"   JSD (Code Frequency): {jsd_code:.4f} (0=identical, 1=different)")
    print(f"   Correlation (Code Frequency): {corr_code:.4f}")
    
    results['code_frequency'] = {
        'jsd': float(jsd_code),
        'correlation': float(corr_code),
        'num_codes_real': int((real_freq > 0).sum()),
        'num_codes_synth': int((synth_freq > 0).sum()),
        'num_codes_both': int(((real_freq > 0) & (synth_freq > 0)).sum())
    }
    
    # 2. Visit Length Distribution
    print("\n2. Visit Length Distribution...")
    real_visit_lens = compute_visit_lengths(real_data)
    synth_visit_lens = compute_visit_lengths(synth_data)
    
    ks_stat, ks_pval = ks_test(real_visit_lens, synth_visit_lens)
    
    print(f"   Real: mean={np.mean(real_visit_lens):.2f}, std={np.std(real_visit_lens):.2f}")
    print(f"   Synth: mean={np.mean(synth_visit_lens):.2f}, std={np.std(synth_visit_lens):.2f}")
    print(f"   KS Statistic: {ks_stat:.4f}, p-value: {ks_pval:.4e}")
    
    results['visit_length'] = {
        'real_mean': float(np.mean(real_visit_lens)),
        'real_std': float(np.std(real_visit_lens)),
        'synth_mean': float(np.mean(synth_visit_lens)),
        'synth_std': float(np.std(synth_visit_lens)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval)
    }
    
    # 3. Patient Length Distribution (visits per patient)
    print("\n3. Patient Length Distribution (visits per patient)...")
    real_patient_lens = compute_patient_lengths(real_data)
    synth_patient_lens = compute_patient_lengths(synth_data)
    
    ks_stat_p, ks_pval_p = ks_test(real_patient_lens, synth_patient_lens)
    
    print(f"   Real: mean={np.mean(real_patient_lens):.2f}, std={np.std(real_patient_lens):.2f}")
    print(f"   Synth: mean={np.mean(synth_patient_lens):.2f}, std={np.std(synth_patient_lens):.2f}")
    print(f"   KS Statistic: {ks_stat_p:.4f}, p-value: {ks_pval_p:.4e}")
    
    results['patient_length'] = {
        'real_mean': float(np.mean(real_patient_lens)),
        'real_std': float(np.std(real_patient_lens)),
        'synth_mean': float(np.mean(synth_patient_lens)),
        'synth_std': float(np.std(synth_patient_lens)),
        'ks_statistic': float(ks_stat_p),
        'ks_pvalue': float(ks_pval_p)
    }
    
    # 4. Code Co-occurrence
    print("\n4. Code Co-occurrence (top 100 codes)...")
    real_cooc, top_codes = compute_code_cooccurrence(real_data, vocab_size, top_k=100)
    synth_cooc, _ = compute_code_cooccurrence(synth_data, vocab_size, top_k=100)
    
    # Flatten and compare
    corr_cooc = correlation(real_cooc, synth_cooc)
    jsd_cooc = jensen_shannon_divergence(real_cooc.flatten(), synth_cooc.flatten())
    
    print(f"   Correlation (Co-occurrence): {corr_cooc:.4f}")
    print(f"   JSD (Co-occurrence): {jsd_cooc:.4f}")
    
    results['code_cooccurrence'] = {
        'correlation': float(corr_cooc),
        'jsd': float(jsd_cooc)
    }
    
    # 5. Label Distribution
    print("\n5. Label Distribution...")
    real_labels = compute_label_distribution(real_data)
    synth_labels = compute_label_distribution(synth_data)
    
    if real_labels is not None and synth_labels is not None:
        jsd_labels = jensen_shannon_divergence(real_labels, synth_labels)
        corr_labels = correlation(real_labels, synth_labels)
        
        print(f"   JSD (Labels): {jsd_labels:.4f}")
        print(f"   Correlation (Labels): {corr_labels:.4f}")
        
        # Per-label comparison
        print("\n   Per-label prevalence (Real vs Synth):")
        for i, (r, s) in enumerate(zip(real_labels, synth_labels)):
            diff = abs(r - s)
            marker = "[!]" if diff > 0.1 else "[ok]"
            print(f"   Label {i:2d}: Real={r:.3f}, Synth={s:.3f}, Diff={diff:.3f} {marker}")
        
        results['label_distribution'] = {
            'jsd': float(jsd_labels),
            'correlation': float(corr_labels),
            'real_prevalence': [float(x) for x in real_labels],
            'synth_prevalence': [float(x) for x in synth_labels]
        }
    else:
        print("   Labels not available in dataset")
        results['label_distribution'] = None
    
    # 6. Temporal Distribution
    print("\n6. Temporal Distribution (code positions across visits)...")
    real_temporal = compute_temporal_distribution(real_data, vocab_size)
    synth_temporal = compute_temporal_distribution(synth_data, vocab_size)
    
    corr_temporal = correlation(real_temporal, synth_temporal)
    
    print(f"   Correlation (Temporal): {corr_temporal:.4f}")
    
    results['temporal_distribution'] = {
        'correlation': float(corr_temporal)
    }
    
    # 7. Summary Statistics
    print("\n" + "-"*70)
    print("Fidelity summary")
    print("-"*70)
    
    # Compute overall fidelity score (average of correlations)
    correlations = [
        results['code_frequency']['correlation'],
        results['code_cooccurrence']['correlation'],
        results['temporal_distribution']['correlation']
    ]
    if results['label_distribution']:
        correlations.append(results['label_distribution']['correlation'])
    
    avg_correlation = np.mean(correlations)
    
    # Compute overall JSD (average)
    jsds = [
        results['code_frequency']['jsd'],
        results['code_cooccurrence']['jsd']
    ]
    if results['label_distribution']:
        jsds.append(results['label_distribution']['jsd'])
    
    avg_jsd = np.mean(jsds)
    
    print(f"\n   Average Correlation: {avg_correlation:.4f}")
    print(f"   Average JSD: {avg_jsd:.4f}")
    
    results['summary'] = {
        'avg_correlation': float(avg_correlation),
        'avg_jsd': float(avg_jsd),
        'num_real_patients': len(real_data),
        'num_synth_patients': len(synth_data),
        'vocab_size': vocab_size
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate fidelity of synthetic EHR data')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory with real data')
    parser.add_argument('--synth_path', type=str, default='./results/datasets/haloDataset.pkl', 
                        help='Path to synthetic dataset')
    parser.add_argument('--output', type=str, default='fidelity_results.json', 
                        help='Output JSON file')
    parser.add_argument('--dataset', type=str, default='unknown', 
                        help='Dataset name (mimic3 or mimic4)')
    
    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Synthetic path: {args.synth_path}")
    
    # Load data
    real_data, synth_data, vocab_size, id_to_code = load_data(args.data_dir, args.synth_path)
    
    # Run evaluation
    results = evaluate_fidelity(real_data, synth_data, vocab_size, id_to_code)
    results['dataset'] = args.dataset
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
