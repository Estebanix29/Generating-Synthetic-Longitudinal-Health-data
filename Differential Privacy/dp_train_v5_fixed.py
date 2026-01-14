"""
# DP Training v5 - fixed vocab size (code_vocab_size=6984, total_vocab_size=7012)

import os
import sys
import math
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import HALOModel
from config import HALOConfig

# RDP privacy accounting (from Opacus)
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

def get_batch(loc, batch_size, mode, dataset, config):
    """
    HALO's custom batching function (from train_model.py)
    Creates one-hot encoded visit tensors with shape (batch, n_ctx, total_vocab_size)
    
    No longer filters codes - all codes 0-6983 are valid
    """
    if mode == 'train':
        ehr = dataset[loc:loc+batch_size]
    elif mode == 'val':
        ehr = dataset[loc:loc+batch_size]
    else:
        raise ValueError("Invalid mode")

    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))

    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_ehr[i, j+2][v] = 1  # One-hot encode visit codes
            batch_mask[i, j+2] = 1
        # Set patient labels
        batch_ehr[i, 1, config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels'])
        # Set end token
        batch_ehr[i, len(visits)+1, config.code_vocab_size+config.label_vocab_size+1] = 1
        # Set padding tokens
        batch_ehr[i, len(visits)+2:, config.code_vocab_size+config.label_vocab_size+2] = 1

    batch_mask[:, 1] = 1  # Mask covers labels
    batch_ehr[:, 0, config.code_vocab_size+config.label_vocab_size] = 1  # Start token
    batch_mask = batch_mask[:, 1:, :]  # Shift mask

    return batch_ehr, batch_mask

def compute_epsilon(steps, batch_size, dataset_size, noise_multiplier, delta):
    """Compute epsilon using RDP accountant."""
    sample_rate = batch_size / dataset_size
    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    epsilon = accountant.get_epsilon(delta)
    return epsilon

def train_with_custom_dp():
    print("Differential privacy training - version 5 (fixed vocab size)")

    
    
    # HYPERPARAMETERS (Same as v4 for comparison)
    
    TARGET_EPSILON = 50.0
    TARGET_DELTA = 1e-5
    MAX_GRAD_NORM = 50.0
    BATCH_SIZE = 512
    LEARNING_RATE = 0.0008
    NUM_EPOCHS = 40
    PATIENCE = 5
    MIN_DELTA_LOSS = 100.0
    
    print(f"Target Privacy: (ε={TARGET_EPSILON:.2f}, δ={TARGET_DELTA})")
    print(f"Max Gradient Norm (C): {MAX_GRAD_NORM}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Early Stopping: Patience={PATIENCE}, Min Δ Loss={MIN_DELTA_LOSS}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Load data
    print("Loading datasets...")
    with open('data/trainDataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('data/valDataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print()
    
    config = HALOConfig()
    config.batch_size = BATCH_SIZE
    
    print(f"Model vocab size: {config.total_vocab_size} (code={config.code_vocab_size}, label={config.label_vocab_size}, special={config.special_vocab_size})")
    print()
    
    n_train = len(train_dataset)
    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    total_steps = steps_per_epoch * NUM_EPOCHS
    sample_rate = BATCH_SIZE / n_train
    
    print("-"*60)
    print("Privacy budget calculation")
    print("-"*60)
    print(f"Training samples (N): {n_train:,}")
    print(f"Batch size (B): {BATCH_SIZE}")
    print(f"Sampling rate (q=B/N): {sample_rate:.6f}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Total steps (T): {total_steps:,}")
    print()
    
    # Compute required noise multiplier
    noise_multiplier = get_noise_multiplier(
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        sample_rate=sample_rate,
        epochs=NUM_EPOCHS,
        accountant='rdp'
    )
    
    print(f"Computed noise multiplier: {noise_multiplier:.4f}")
    print()
    
    # Verify epsilon
    final_epsilon = compute_epsilon(total_steps, BATCH_SIZE, n_train, noise_multiplier, TARGET_DELTA)
    print(f"Privacy guarantee: (eps={final_epsilon:.2f}, delta={TARGET_DELTA})")
    print()
    
    # Model initialization
    print("-"*60)
    print("Model initialization")
    print("-"*60)
    
    model = HALOModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    print(f"Embedding input size: {model.transformer.vis_embed_mat.in_features}")
    print()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Optimizer: Adam")
    print(f"Learning rate: {LEARNING_RATE}")
    print()
    
    # Training loop
    checkpoint_dir = 'checkpoints/custom_dp_v5'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save training log
    log_file = os.path.join(checkpoint_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Custom DP-SGD Training Started: {datetime.now()}\n")
        f.write(f"Privacy: (ε={TARGET_EPSILON}, δ={TARGET_DELTA})\n")
        f.write(f"Noise multiplier: {noise_multiplier:.4f}\n")
        f.write(f"Vocab size fixed: code={config.code_vocab_size}, total={config.total_vocab_size}\n")
        f.write("\n\n")
    
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    current_step = 0
    
    print("-"*60)
    print("Training started")
    print("-"*60)
    print()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norms = []
        epoch_clipped_count = 0
        num_batches = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(train_dataset))
        shuffled_dataset = [train_dataset[i] for i in indices]
        
        progress_bar = tqdm(range(0, len(shuffled_dataset), BATCH_SIZE), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for i in progress_bar:
            actual_batch_size = min(BATCH_SIZE, len(shuffled_dataset) - i)
            
            # Get batch using HALO's custom function
            batch_ehr, batch_mask = get_batch(i, actual_batch_size, 'train', shuffled_dataset, config)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            
            # Forward pass
            loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, 
                              ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # DP: Gradient clipping + noise
            grad_norm = torch.sqrt(sum(
                p.grad.norm() ** 2 for p in model.parameters() 
                if p.grad is not None
            ))
            
            epoch_grad_norms.append(grad_norm.item())
            
            # 2. Clip gradient
            if grad_norm > MAX_GRAD_NORM:
                clip_factor = MAX_GRAD_NORM / grad_norm
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clip_factor)
                epoch_clipped_count += 1
            
            # 3. Add Gaussian noise: N(0, σ²C²/n²)
            noise_std = MAX_GRAD_NORM * noise_multiplier / BATCH_SIZE
            
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_std,
                        size=p.grad.shape,
                        device=p.grad.device
                    )
                    p.grad.add_(noise)
            
            # Optimizer step
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            current_step += 1
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'grad': f'{grad_norm.item():.2f}'
            })
        
        # Epoch complete: validation
        avg_train_loss = epoch_loss / num_batches
        avg_grad_norm = np.mean(epoch_grad_norms)
        clipping_rate = epoch_clipped_count / num_batches
        
        print()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Gradient Norm: {avg_grad_norm:.4f}")
        print(f"  Clipping Rate: {clipping_rate*100:.2f}% ({epoch_clipped_count}/{num_batches} batches)")
        
        # Compute current privacy spent
        steps_so_far = (epoch + 1) * steps_per_epoch
        current_epsilon = compute_epsilon(steps_so_far, BATCH_SIZE, n_train, noise_multiplier, TARGET_DELTA)
        print(f"  Privacy ε: {current_epsilon:.2f}/{TARGET_EPSILON:.2f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_dataset), config.batch_size):
                batch_ehr, batch_mask = get_batch(i, min(config.batch_size, len(val_dataset)-i), 
                                                  'val', val_dataset, config)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
                
                loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr,
                                  ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / max(num_val_batches, 1)
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"\nEpoch {epoch+1}:\n")
            f.write(f"  Train Loss: {avg_train_loss:.4f}\n")
            f.write(f"  Val Loss: {avg_val_loss:.4f}\n")
            f.write(f"  Privacy ε: {current_epsilon:.4f}/{TARGET_EPSILON:.2f}\n")
            f.write(f"  Gradient Norm: {avg_grad_norm:.4f}\n")
            f.write(f"  Clipping Rate: {clipping_rate*100:.2f}%\n")
        
        # Early stopping based on validation loss
        if avg_val_loss < best_loss - MIN_DELTA_LOSS:
            print(f"  New best loss: {best_loss:.4f} -> {avg_val_loss:.4f} (delta={best_loss-avg_val_loss:.2f})")
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'privacy_spent': current_epsilon,
                'config': {
                    'total_vocab_size': config.total_vocab_size,
                    'code_vocab_size': config.code_vocab_size,
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            if epochs_without_improvement >= PATIENCE:
                print()
                print("-"*60)
                print(f"Early stopping: no improvement for {PATIENCE} epochs")
                print("-"*60)
                break
        
        print()
    
    # Training complete
    final_epsilon = compute_epsilon(steps_so_far, BATCH_SIZE, n_train, noise_multiplier, TARGET_DELTA)
    
    print()
    print("-"*60)
    print("Training complete")
    print("-"*60)
    print(f"Best Loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"Final Privacy: (ε={final_epsilon:.2f}, δ={TARGET_DELTA})")
    print(f"Model saved: {checkpoint_dir}/best_model.pth")
    print(f"Log saved: {log_file}")
    print()

if __name__ == '__main__':
    train_with_custom_dp()
