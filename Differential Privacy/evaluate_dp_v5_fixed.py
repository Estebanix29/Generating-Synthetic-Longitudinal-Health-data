"""
import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model import HALOModel
from config import HALOConfig

def get_batch(loc, batch_size, dataset, config):
    ehr = dataset[loc:loc+batch_size]
    
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))

    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_ehr[i, j+2][v] = 1
            batch_mask[i, j+2] = 1
        batch_ehr[i, 1, config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels'])
        batch_ehr[i, len(visits)+1, config.code_vocab_size+config.label_vocab_size+1] = 1
        batch_ehr[i, len(visits)+2:, config.code_vocab_size+config.label_vocab_size+2] = 1

    batch_mask[:, 1] = 1
    batch_ehr[:, 0, config.code_vocab_size+config.label_vocab_size] = 1
    batch_mask = batch_mask[:, 1:, :]

    return batch_ehr, batch_mask

def evaluate_model():
    print("-"*60)
    print("DP-HALO v5 evaluation (fixed vocab size)")
    print("-"*60)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load config and model
    config = HALOConfig()
    print(f"Config vocab size: {config.total_vocab_size} (code={config.code_vocab_size})")
    
    model = HALOModel(config).to(device)
    
    checkpoint_path = 'checkpoints/custom_dp_v5/best_model.pth'
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Privacy ε: {checkpoint['privacy_spent']:.4f}")
    print()
    
    # Load test data
    with open('data/testDataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    
    with open('data/idToLabel.pkl', 'rb') as f:
        id_to_label = pickle.load(f)
    
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Number of labels: {len(id_to_label)}")
    print()
    
    # Evaluate - use the model's forward pass properly
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for i in range(0, len(test_dataset), config.batch_size):
            batch_size = min(config.batch_size, len(test_dataset)-i)
            batch_ehr, batch_mask = get_batch(i, batch_size, test_dataset, config)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            
            # Forward pass (same as training)
            loss, predictions, labels = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, 
                                             ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
            
            # DEBUG: Check shapes
            if i == 0:
                print(f"DEBUG: predictions shape = {predictions.shape}")
                print(f"DEBUG: labels shape = {labels.shape}")
                print(f"DEBUG: batch_ehr shape = {batch_ehr.shape}")
                print(f"DEBUG: config.code_vocab_size = {config.code_vocab_size}")
                print(f"DEBUG: config.label_vocab_size = {config.label_vocab_size}")
            
            # KEY INSIGHT: predictions is [batch, seq-1, code_vocab_size]
            # It only predicts CODES (0-6983), not labels/special tokens
            # Labels are in the INPUT at position 1: batch_ehr[:, 1, 6984:7009]
            # Position 0 of predictions predicts what should be at position 1 of input
            # But we can't extract labels from predictions since it doesn't include them!
            
            # SOLUTION: Get true labels from input, compare against predictions at position 0
            # But predictions doesn't have label dimensions... 
            # This means the model CAN'T predict labels - it only predicts codes!
            
            # Let me check if predictions actually has full vocab despite the slicing
            if i == 0:
                print(f"DEBUG: Checking position 0 predictions...")
                print(f"DEBUG: predictions[0, 0, :10] = {predictions[0, 0, :10]}")
                print(f"DEBUG: predictions min/max = {predictions.min()}/{predictions.max()}")
            
            batch_labels = batch_ehr[:, 1, config.code_vocab_size:config.code_vocab_size+config.label_vocab_size].cpu().numpy()
            preds = np.zeros_like(batch_labels)  # Placeholder - need to figure out correct extraction
            
            # Get true labels
            batch_labels = np.array([p['labels'] for p in test_dataset[i:i+batch_size]])
            
            all_preds.append(preds)
            all_labels.append(batch_labels)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    print(f"Predictions shape: {all_preds.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print()
    
    # Compute metrics
    print("-"*60)
    print("Results")
    print("-"*60)
    
    # Overall metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Accuracy (exact match)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\nOverall Metrics:")
    print(f"  Micro F1:      {f1_micro:.4f}")
    print(f"  Macro F1:      {f1_macro:.4f}")
    print(f"  Weighted F1:   {f1_weighted:.4f}")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  Precision (μ): {precision_micro:.4f}")
    print(f"  Recall (μ):    {recall_micro:.4f}")
    
    # Per-label metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\nPer-Label F1 Scores:")
    for i, label_name in enumerate(id_to_label):
        print(f"  {label_name:50s} F1={f1_per[i]:.4f} P={precision_per[i]:.4f} R={recall_per[i]:.4f} (n={int(support_per[i])})")
    
    # Prediction behavior analysis
    print(f"\nPrediction Behavior:")
    avg_preds_per_sample = all_preds.sum(axis=1).mean()
    avg_labels_per_sample = all_labels.sum(axis=1).mean()
    all_positive_rate = (all_preds.sum(axis=1) == len(id_to_label)).mean()
    all_negative_rate = (all_preds.sum(axis=1) == 0).mean()
    
    print(f"  Avg predictions per sample: {avg_preds_per_sample:.2f}")
    print(f"  Avg true labels per sample: {avg_labels_per_sample:.2f}")
    print(f"  All-positive rate: {all_positive_rate*100:.2f}%")
    print(f"  All-negative rate: {all_negative_rate*100:.2f}%")
    
    # Save results
    results = {
        'model_info': {
            'version': 'v5',
            'vocab_fix': 'Applied (code_vocab=6984, total_vocab=7012)',
            'model_path': checkpoint_path,
            'epoch': int(checkpoint['epoch']),
            'val_loss': float(checkpoint['val_loss']),
            'epsilon': float(checkpoint['privacy_spent']),
            'delta': 1e-5,
        },
        'dataset': {
            'num_samples': len(test_dataset),
            'num_labels': len(id_to_label),
        },
        'metrics': {
            'accuracy': float(accuracy),
            'per_label': {
                'precision': precision_per.tolist(),
                'recall': recall_per.tolist(),
                'f1': f1_per.tolist(),
                'support': support_per.tolist(),
            },
            'micro': {
                'precision': float(precision_micro),
                'recall': float(recall_micro),
                'f1': float(f1_micro),
            },
            'macro': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1': float(f1_macro),
            },
            'weighted': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1': float(f1_weighted),
            },
        },
        'analysis': {
            'avg_predictions_per_sample': float(avg_preds_per_sample),
            'avg_true_labels_per_sample': float(avg_labels_per_sample),
            'all_positive_rate': float(all_positive_rate),
            'all_negative_rate': float(all_negative_rate),
        }
    }
    
    output_file = 'checkpoints/custom_dp_v5/evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print()

if __name__ == '__main__':
    evaluate_model()
