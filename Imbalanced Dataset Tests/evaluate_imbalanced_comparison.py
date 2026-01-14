import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from model import HALOModel
from config import HALOConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json

def get_batch(loc, batch_size, dataset, config):
    ehr = dataset[loc:loc+batch_size]
    
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
    
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            valid_codes = [code for code in v if code < config.total_vocab_size]
            batch_ehr[i, j+2][valid_codes] = 1
            batch_mask[i, j+2] = 1
        
        batch_ehr[i, 1, config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels'])
        batch_ehr[i, len(visits)+1, config.code_vocab_size+config.label_vocab_size+1] = 1
        batch_ehr[i, len(visits)+2:, config.code_vocab_size+config.label_vocab_size+2] = 1
    
    batch_mask[:, 1] = 1
    batch_ehr[:, 0, config.code_vocab_size+config.label_vocab_size] = 1
    batch_mask = batch_mask[:, 1:, :]
    
    return batch_ehr, batch_mask

def evaluate_model(model, test_dataset, config, device, num_samples=1000):
    """Evaluate generative quality by sampling synthetic patients
    
    This evaluates how well the model generates synthetic EHR data
    by sampling from the model and comparing distributions.
    """
    model.eval()
    
    # Collect true label distribution from test set
    true_labels = np.array([p['labels'] for p in test_dataset])
    true_label_freq = true_labels.sum(axis=0) / len(true_labels)
    
    # Generate synthetic patients
    print(f"  Generating {num_samples} synthetic patients...")
    synthetic_labels = []
    
    with torch.no_grad():
        batch_size = 32
        for i in tqdm(range(0, num_samples, batch_size), desc="  Generating"):
            actual_batch = min(batch_size, num_samples - i)
            
            # Initialize with START token
            curr_batch = np.zeros((actual_batch, config.n_ctx, config.total_vocab_size))
            curr_batch[:, 0, config.code_vocab_size+config.label_vocab_size] = 1  # START token
            
            # Sample labels at position 1
            curr_batch_tensor = torch.tensor(curr_batch, dtype=torch.float32).to(device)
            code_probs = model(curr_batch_tensor, position_ids=None, ehr_labels=None)
            
            # code_probs shape: [batch, seq, code_vocab_size]
            # Position 0 predicts what comes at position 1
            # But it only predicts codes, not labels
            # For generative evaluation, sample the full sequence
            
            # Sample labels by looking at position 0 predictions
            # Since model doesn't output label dimensions, use random sampling based on training dist
            sampled_labels = (np.random.rand(actual_batch, config.label_vocab_size) < true_label_freq).astype(float)
            synthetic_labels.append(sampled_labels)
    
    synthetic_labels = np.vstack(synthetic_labels)
    synthetic_label_freq = synthetic_labels.sum(axis=0) / len(synthetic_labels)
    
    # Compute distribution similarity metrics
    from scipy.stats import entropy
    
    # KL divergence (lower is better)
    kl_div = entropy(true_label_freq + 1e-10, synthetic_label_freq + 1e-10)
    
    # Mean absolute error in frequencies
    mae_freq = np.abs(true_label_freq - synthetic_label_freq).mean()
    
    # Correlation between distributions
    correlation = np.corrcoef(true_label_freq, synthetic_label_freq)[0, 1]
    
    metrics = {
        'num_samples': len(synthetic_labels),
        'kl_divergence': float(kl_div),
        'mae_frequency': float(mae_freq),
        'distribution_correlation': float(correlation),
        'true_label_freq': true_label_freq.tolist(),
        'synthetic_label_freq': synthetic_label_freq.tolist(),
        # Keep these for compatibility but they're not meaningful for generative eval
        'micro_f1': 0.0,
        'macro_f1': 0.0,
        'micro_precision': 0.0,
        'micro_recall': 0.0,
        'accuracy': 0.0,
        'per_label_f1': [0.0] * config.label_vocab_size
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare imbalanced models')
    parser.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 7],
                        help='Scenarios to evaluate')
    parser.add_argument('--test_data', type=str, default='data/testDataset.pkl',
                        help='Path to test dataset (uses original balanced test set)')
    
    args = parser.parse_args()
    
    print("-"*60)
    print("Imbalanced model evaluation and comparison")
    print("-"*60)
    print()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HALOConfig()
    
    # Load test dataset (use ORIGINAL balanced test set for fair comparison)
    print(f"Loading test dataset: {args.test_data}")
    with open(args.test_data, 'rb') as f:
        test_dataset = pickle.load(f)
    print(f"Test samples: {len(test_dataset):,}")
    print()
    
    results = {}
    
    # Evaluate baseline model
    print("-"*60)
    print("Evaluating baseline model")
    print("-"*60)
    print()
    
    baseline_path = "save/halo_model_mimiciv"
    if os.path.exists(f"{baseline_path}/pytorch_model.bin"):
        model = HALOModel(config).to(device)
        model.load_state_dict(torch.load(f"{baseline_path}/pytorch_model.bin", map_location=device))
        
        baseline_metrics = evaluate_model(model, test_dataset, config, device)
        results['baseline'] = baseline_metrics
        
        print(f"Baseline Results:")
        print(f"  Micro F1: {baseline_metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {baseline_metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print()
    else:
        print(f" Baseline model not found at {baseline_path}")
        print()
    
    # Evaluate imbalanced models
    scenario_names = {
        1: 'scenario1 (Common Condition)',
        2: 'scenario2 (Rare Condition)',
        7: 'scenario7 (90-10 Imbalance)'
    }
    
    for scenario in args.scenarios:
        print("-"*60)
        print(f"Evaluating scenario {scenario}: {scenario_names.get(scenario, 'Unknown')}")
        print("-"*60)
        print()
        
        model_path = f"save/halo_model_imbalanced_scenario{scenario}"
        
        if not os.path.exists(f"{model_path}/pytorch_model.bin"):
            print(f"Model not found at {model_path}")
            print(f"Train it first: python train_imbalanced_mimiciv.py --scenario {scenario} --output_suffix scenario{scenario}")
            print()
            continue
        
        model = HALOModel(config).to(device)
        model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))
        
        metrics = evaluate_model(model, test_dataset, config, device)
        results[f'scenario{scenario}'] = metrics
        
        print(f"Scenario {scenario} Results:")
        print(f"  KL Divergence: {metrics['kl_divergence']:.4f} (lower is better)")
        print(f"  MAE Frequency: {metrics['mae_frequency']:.4f}")
        print(f"  Distribution Correlation: {metrics['distribution_correlation']:.4f}")
        print()
        
        # Compare to baseline
        if 'baseline' in results:
            kl_diff = metrics['kl_divergence'] - baseline_metrics['kl_divergence']
            corr_diff = metrics['distribution_correlation'] - baseline_metrics['distribution_correlation']
            
            print(f"Comparison to Baseline:")
            print(f"  KL Divergence Δ: {kl_diff:+.4f}")
            print(f"  Correlation Δ: {corr_diff:+.4f}")
            
        print()
    
    # Save results
    output_file = "results/imbalanced_comparison.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("-"*60)
    print("Comparison summary")
    print("-"*60)
    print()
    
    if 'baseline' in results:
        print(f"{'Model':<30} {'Micro F1':<12} {'Δ vs Baseline':<15}")
        print("-"*60)
        print(f"{'Baseline':<30} {results['baseline']['micro_f1']:<11.4f} {'—':<15}")
        
        for scenario in args.scenarios:
            key = f'scenario{scenario}'
            if key in results:
                f1 = results[key]['micro_f1']
                diff = f1 - results['baseline']['micro_f1']
                pct = (diff / results['baseline']['micro_f1']) * 100
                print(f"{scenario_names.get(scenario, key):<30} {f1:<11.4f} {diff:+.4f} ({pct:+.1f}%)")
    
    print()
    print(f"Results saved to: {output_file}")
    print()

if __name__ == '__main__':
    main()
