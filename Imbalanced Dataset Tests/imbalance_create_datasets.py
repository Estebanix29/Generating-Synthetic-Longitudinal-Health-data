import pickle
import numpy as np
import argparse
from collections import Counter

def load_datasets():
    print("Loading original datasets...")
    
    with open('data/trainDataset.pkl', 'rb') as f:
        train = pickle.load(f)
    with open('data/valDataset.pkl', 'rb') as f:
        val = pickle.load(f)
    with open('data/testDataset.pkl', 'rb') as f:
        test = pickle.load(f)
    
    print(f"  Train: {len(train):,}")
    print(f"  Val: {len(val):,}")
    print(f"  Test: {len(test):,}")
    print()
    
    return train, val, test

def filter_single_label(dataset, target_label):
    filtered = [s for s in dataset if s['labels'][target_label] == 1]
    return filtered

def filter_multi_label(dataset, target_labels):
    filtered = [
        s for s in dataset 
        if any(s['labels'][label] == 1 for label in target_labels)
    ]
    return filtered

def filter_trajectory_length(dataset, threshold, direction='short'):
    if direction == 'short':
        filtered = [s for s in dataset if len(s['visits']) <= threshold]
    else:  # long
        filtered = [s for s in dataset if len(s['visits']) >= threshold]
    return filtered

def create_oversample_imbalance(dataset, target_label, target_ratio=0.9):
    """
    Create severe imbalance by oversampling one condition to target_ratio.
    
    Strategy:
    - Keep all samples without target label (10%)
    - Oversample samples with target label to 90%
    """
    with_label = [s for s in dataset if s['labels'][target_label] == 1]
    without_label = [s for s in dataset if s['labels'][target_label] == 0]
    
    # Calculate how many with_label samples are needed
    target_total = len(dataset)
    n_without = int(target_total * (1 - target_ratio))
    n_with = int(target_total * target_ratio)
    
    # Sample without_label to get n_without samples
    if len(without_label) > n_without:
        without_label_sampled = np.random.choice(len(without_label), n_without, replace=False)
        without_label = [without_label[i] for i in without_label_sampled]
    
    # Oversample with_label to get n_with samples
    if len(with_label) < n_with:
        # Need to oversample
        with_label_indices = np.random.choice(len(with_label), n_with, replace=True)
        with_label = [with_label[i] for i in with_label_indices]
    else:
        # Subsample
        with_label_sampled = np.random.choice(len(with_label), n_with, replace=False)
        with_label = [with_label[i] for i in with_label_sampled]
    
    # Combine and shuffle
    combined = with_label + without_label
    np.random.shuffle(combined)
    
    return combined

def analyze_dataset_balance(dataset, scenario_name):
    """Analyze label distribution in created dataset."""
    print(f"\nDataset Analysis: {scenario_name}")
    print("-" * 60)
    
    label_counts = np.zeros(25)
    for sample in dataset:
        label_counts += np.array(sample['labels'])
    
    total = len(dataset)
    print(f"Total samples: {total:,}")
    print(f"\nLabel distribution:")
    print(f"{'Label':<8} {'Count':<10} {'Percentage':<12}")
    print("-" * 40)
    
    for label_idx in range(25):
        if label_counts[label_idx] > 0:
            count = int(label_counts[label_idx])
            pct = (count / total) * 100
            print(f"{label_idx:<8} {count:<10,} {pct:<11.2f}%")
    
    # Visit length stats
    visit_lengths = [len(s['visits']) for s in dataset]
    print(f"\nTrajectory statistics:")
    print(f"  Mean visit length: {np.mean(visit_lengths):.2f}")
    print(f"  Median visit length: {np.median(visit_lengths):.0f}")
    print(f"  Min-Max: {np.min(visit_lengths)}-{np.max(visit_lengths)}")
    print()

def create_scenario(train, val, test, scenario_num):
    """Create imbalanced datasets for a specific scenario."""
    
    print("-"*60)
    print(f"Creating imbalanced dataset - scenario {scenario_num}")
    print("-"*60)
    print()
    
    if scenario_num == 1:
        # Dominant Common Condition (Label 0 is usually most common)
        print("Scenario 1: Dominant Common Condition")
        print("Filtering for samples with most common label...")
        
        # First, find most common label
        label_counts = np.zeros(25)
        for s in train:
            label_counts += np.array(s['labels'])
        most_common_label = int(np.argmax(label_counts))
        
        print(f"Most common label: {most_common_label}")
        print()
        
        train_imb = filter_single_label(train, most_common_label)
        val_imb = filter_single_label(val, most_common_label)
        test_imb = filter_single_label(test, most_common_label)
        
        scenario_name = f"scenario1_common_label{most_common_label}"
    
    elif scenario_num == 2:
        # Rare Condition Only
        print("Scenario 2: Rare Condition Only")
        print("Filtering for samples with rarest label...")
        
        # Find rarest label
        label_counts = np.zeros(25)
        for s in train:
            label_counts += np.array(s['labels'])
        rarest_label = int(np.argmin(label_counts[label_counts > 0]))  # Exclude zero counts
        
        print(f"Rarest label: {rarest_label}")
        print()
        
        train_imb = filter_single_label(train, rarest_label)
        val_imb = filter_single_label(val, rarest_label)
        test_imb = filter_single_label(test, rarest_label)
        
        scenario_name = f"scenario2_rare_label{rarest_label}"
    
    elif scenario_num == 3:
        # Top 3 common conditions
        print("Scenario 3: Top 3 Common Conditions")
        
        label_counts = np.zeros(25)
        for s in train:
            label_counts += np.array(s['labels'])
        top3_labels = np.argsort(label_counts)[-3:].tolist()
        
        print(f"Top 3 labels: {top3_labels}")
        print()
        
        train_imb = filter_multi_label(train, top3_labels)
        val_imb = filter_multi_label(val, top3_labels)
        test_imb = filter_multi_label(test, top3_labels)
        
        scenario_name = f"scenario3_top3_labels"
    
    elif scenario_num == 4:
        # Bottom 5 rare conditions
        print("Scenario 4: Bottom 5 Rare Conditions")
        
        label_counts = np.zeros(25)
        for s in train:
            label_counts += np.array(s['labels'])
        
        # Get bottom 5 (excluding any zeros)
        non_zero_indices = np.where(label_counts > 0)[0]
        sorted_indices = non_zero_indices[np.argsort(label_counts[non_zero_indices])]
        bottom5_labels = sorted_indices[:5].tolist()
        
        print(f"Bottom 5 labels: {bottom5_labels}")
        print()
        
        train_imb = filter_multi_label(train, bottom5_labels)
        val_imb = filter_multi_label(val, bottom5_labels)
        test_imb = filter_multi_label(test, bottom5_labels)
        
        scenario_name = f"scenario4_bottom5_labels"
    
    elif scenario_num == 5:
        # Short trajectories only
        print("Scenario 5: Short Trajectories Only")
        
        visit_lengths = [len(s['visits']) for s in train]
        threshold = int(np.percentile(visit_lengths, 25))
        
        print(f"Threshold (25th percentile): ≤{threshold} visits")
        print()
        
        train_imb = filter_trajectory_length(train, threshold, 'short')
        val_imb = filter_trajectory_length(val, threshold, 'short')
        test_imb = filter_trajectory_length(test, threshold, 'short')
        
        scenario_name = f"scenario5_short_traj_le{threshold}"
    
    elif scenario_num == 6:
        # Long trajectories only
        print("Scenario 6: Long Trajectories Only")
        
        visit_lengths = [len(s['visits']) for s in train]
        threshold = int(np.percentile(visit_lengths, 75))
        
        print(f"Threshold (75th percentile): ≥{threshold} visits")
        print()
        
        train_imb = filter_trajectory_length(train, threshold, 'long')
        val_imb = filter_trajectory_length(val, threshold, 'long')
        test_imb = filter_trajectory_length(test, threshold, 'long')
        
        scenario_name = f"scenario6_long_traj_ge{threshold}"
    
    elif scenario_num == 7:
        # 90-10 severe imbalance
        print("Scenario 7: 90-10 Artificial Imbalance")
        
        label_counts = np.zeros(25)
        for s in train:
            label_counts += np.array(s['labels'])
        
        # Choose middle-frequency label
        sorted_labels = np.argsort(label_counts)
        mid_label = int(sorted_labels[len(sorted_labels)//2])
        
        print(f"Target label for 90% dominance: {mid_label}")
        print()
        
        train_imb = create_oversample_imbalance(train, mid_label, 0.9)
        val_imb = create_oversample_imbalance(val, mid_label, 0.9)
        test_imb = create_oversample_imbalance(test, mid_label, 0.9)
        
        scenario_name = f"scenario7_90-10_label{mid_label}"
    
    else:
        raise ValueError(f"Invalid scenario number: {scenario_num}. Must be 1-7.")
    
    # Analyze created datasets
    analyze_dataset_balance(train_imb, f"{scenario_name} - Train")
    analyze_dataset_balance(val_imb, f"{scenario_name} - Val")
    analyze_dataset_balance(test_imb, f"{scenario_name} - Test")
    
    # Save datasets
    print("-"*60)
    print("Saving imbalanced datasets")
    print("-"*60)
    print()
    
    import os
    output_dir = f"data/imbalanced/{scenario_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/trainDataset.pkl", 'wb') as f:
        pickle.dump(train_imb, f)
    print(f"Saved: {output_dir}/trainDataset.pkl ({len(train_imb):,} samples)")
    
    with open(f"{output_dir}/valDataset.pkl", 'wb') as f:
        pickle.dump(val_imb, f)
    print(f"Saved: {output_dir}/valDataset.pkl ({len(val_imb):,} samples)")
    
    with open(f"{output_dir}/testDataset.pkl", 'wb') as f:
        pickle.dump(test_imb, f)
    print(f"Saved: {output_dir}/testDataset.pkl ({len(test_imb):,} samples)")
    
    print()
    print("-"*60)
    print("Dataset creation complete")
    print("-"*60)
    print()
    print(f"Imbalanced datasets saved to: {output_dir}/")
    print()
    print("Next steps:")
    print(f"1. Train baseline model on imbalanced data:")
    print(f"   python train_imbalanced_model.py --data_dir {output_dir}")
    print(f"2. Evaluate and compare to original model")
    print()

def main():
    parser = argparse.ArgumentParser(description='Create imbalanced MIMIC-IV datasets')
    parser.add_argument('--scenario', type=int, required=True, choices=range(1, 8),
                        help='Scenario number (1-7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load datasets
    train, val, test = load_datasets()
    
    # Create scenario
    create_scenario(train, val, test, args.scenario)

if __name__ == '__main__':
    main()
