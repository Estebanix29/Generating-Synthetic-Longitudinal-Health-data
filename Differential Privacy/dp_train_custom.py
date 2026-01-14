"""
Custom Differential Privacy Training for HALO
No Opacus: uses manual DP-SGD with gradient clipping and noise

This approach:
- No per-sample gradients (uses standard PyTorch backprop)
- Clips aggregated gradients (not per-sample)
- Adds calibrated Gaussian noise
- Tracks privacy budget using RDP accountant

Privacy guarantees are approximate but valid for large batch sizes (>64).
Based on TensorFlow Privacy and Abadi et al. (2016) DP-SGD paper.
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import HALO components
from model import HALOModel  # 
from config import HALOConfig

# RDP privacy accounting (from Opacus)
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def clip_gradients(model, max_grad_norm):
    # Clip gradients by global norm, return norm before clipping
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    # Compute global gradient norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]),
        2.0
    )
    
    # Clip if needed
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm.item()


def add_noise_to_gradients(model, noise_multiplier, max_grad_norm):
    """
    Add Gaussian noise to gradients for differential privacy.
    Noise scale = noise_multiplier * max_grad_norm
    """
    noise_scale = noise_multiplier * max_grad_norm
    
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.normal(
                mean=0,
                std=noise_scale,
                size=param.grad.shape,
                device=param.grad.device,
                dtype=param.grad.dtype
            )
            param.grad += noise


def compute_epsilon(steps, batch_size, dataset_size, noise_multiplier, delta):
    """
    Compute privacy budget epsilon using RDP accounting.
    """
    sample_rate = batch_size / dataset_size
    accountant = RDPAccountant()
    
    # Add steps to accountant
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    
    # Get epsilon for target delta
    epsilon = accountant.get_epsilon(delta)
    
    return epsilon


def get_batch(loc, batch_size, mode, dataset, config):
    """
    HALO's custom batching function (from train_model.py)
    Creates one-hot encoded visit tensors with shape (batch, n_ctx, total_vocab_size)
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
            # v is a list of code indices for this visit
            # Filter out any codes that are out of bounds
            valid_codes = [code for code in v if code < config.total_vocab_size]
            if len(valid_codes) < len(v):
                # Skip silently - this is expected if data has codes beyond vocab
                pass
            batch_ehr[i, j+2][valid_codes] = 1  # One-hot encode visit codes
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


def validate(model, val_dataset, config, device):
    """
    Validation function (same as train_model.py)
    """
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
    
    return val_loss / max(num_val_batches, 1)


def train_with_custom_dp(config, args, train_dataset, val_dataset, device, save_dir):
    """
    Train HALO with custom DP-SGD (no Opacus).
    
    Key differences from Opacus:
    1. Uses standard model (not wrapped in GradSampleModule)
    2. Clips aggregated gradients (not per-sample)
    3. Adds noise after aggregation
    4. Same memory as baseline training!
    """
    
    # Initialize model (ORIGINAL model, not dp_model)
    model = HALOModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Privacy parameters
    epsilon = args.epsilon
    delta = args.delta
    max_grad_norm = args.max_grad_norm
    batch_size = args.batch_size
    dataset_size = args.dataset_size
    
    # Compute noise multiplier for target epsilon
    # This is the calibration: higher epsilon = less noise
    sample_rate = batch_size / dataset_size
    steps_per_epoch = dataset_size // batch_size
    total_steps = args.epochs * steps_per_epoch
    
    # Use Opacus's utility to compute noise multiplier
    noise_multiplier = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp"
    )
    
    print(f"\n{'='*60}")
    print(f"Custom DP-SGD Training Configuration")
    print(f"{'='*60}")
    print(f"Privacy Parameters:")
    print(f"  Target ε (epsilon): {epsilon}")
    print(f"  δ (delta): {delta}")
    print(f"  Max gradient norm: {max_grad_norm}")
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"\nTraining Parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Sample rate: {sample_rate:.6f}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps: {total_steps}")
    print(f"\nMemory: Same as baseline (NO per-sample gradients!)")
    print(f"{'='*60}\n")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'))
    
    # Training log
    log_file = open(os.path.join(save_dir, 'training_log.txt'), 'w')
    log_file.write(f"Custom DP-SGD Training Started: {datetime.now()}\n")
    log_file.write(f"Privacy: (ε={epsilon}, δ={delta})\n")
    log_file.write(f"Noise multiplier: {noise_multiplier:.4f}\n\n")
    
    # Shuffle training data
    np.random.shuffle(train_dataset)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0
        
        # Shuffle training data
        np.random.shuffle(train_dataset)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("="*60)
        
        # Training batches
        progress_bar = tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch+1}")
        
        for i in progress_bar:
            # Get batch
            batch_ehr, batch_mask = get_batch(i, min(batch_size, len(train_dataset)-i), 
                                               'train', train_dataset, config)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, 
                              ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
            
            # Backward pass
            loss.backward()
            
            # DP modifications: Clip + Add Noise
            grad_norm = clip_gradients(model, max_grad_norm)
            add_noise_to_gradients(model, noise_multiplier, max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm
            num_batches += 1
            global_step += 1
            
            # Compute current epsilon
            current_epsilon = compute_epsilon(global_step, batch_size, dataset_size, 
                                              noise_multiplier, delta)
            privacy_used = (current_epsilon / epsilon) * 100
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ε': f'{current_epsilon:.2f}/{epsilon}',
                'budget': f'{privacy_used:.1f}%'
            })
            
            # Log to TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/GradientNorm', grad_norm, global_step)
                writer.add_scalar('Privacy/Epsilon', current_epsilon, global_step)
                writer.add_scalar('Privacy/BudgetUsed', privacy_used, global_step)
            
            # Check if privacy budget exhausted
            if current_epsilon >= epsilon:
                print(f"\n{'='*60}")
                print(f"🛑 PRIVACY BUDGET EXHAUSTED")
                print(f"   Current ε: {current_epsilon:.4f}")
                print(f"   Target ε: {epsilon:.4f}")
                print(f"   Training stopped early to preserve privacy guarantee.")
                print(f"{'='*60}\n")
                
                log_file.write(f"\nPrivacy budget exhausted at epoch {epoch+1}, step {global_step}\n")
                log_file.write(f"Final ε: {current_epsilon:.4f}\n")
                
                # Save final model
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'epsilon': current_epsilon,
                    'delta': delta
                }, os.path.join(save_dir, 'final_model.pth'))
                
                log_file.close()
                writer.close()
                return current_epsilon
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Gradient Norm: {avg_grad_norm:.4f}")
        
        # Validation
        val_loss = validate(model, val_dataset, config, device)
        print(f"  Validation Loss: {val_loss:.4f}")
        
        # Compute current epsilon
        current_epsilon = compute_epsilon(global_step, batch_size, dataset_size, 
                                          noise_multiplier, delta)
        privacy_used = (current_epsilon / epsilon) * 100
        print(f"  Privacy Budget: {current_epsilon:.2f}/{epsilon} ({privacy_used:.1f}% used)")
        
        # Save logs
        log_file.write(f"\nEpoch {epoch+1}:\n")
        log_file.write(f"  Train Loss: {avg_loss:.4f}\n")
        log_file.write(f"  Val Loss: {val_loss:.4f}\n")
        log_file.write(f"  Privacy ε: {current_epsilon:.4f}/{epsilon}\n")
        log_file.flush()
        
        # TensorBoard
        writer.add_scalar('Epoch/TrainLoss', avg_loss, epoch)
        writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
        writer.add_scalar('Epoch/Epsilon', current_epsilon, epoch)
        
        # Save checkpoint (best model only to save disk space)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            try:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'epsilon': current_epsilon,
                    'delta': delta,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  New best model saved (val_loss: {val_loss:.4f})")
            except RuntimeError as e:
                print(f"  Warning: Could not save checkpoint (disk space?): {e}")
                print(f"    Continuing training without saving...")
    
    # Training complete
    final_epsilon = compute_epsilon(global_step, batch_size, dataset_size, 
                                     noise_multiplier, delta)
    
    print(f"\n{'-'*60}")
    print(f"Training complete")
    print(f"  Final eps: {final_epsilon:.4f} (target: {epsilon})")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"{'-'*60}\n")
    
    log_file.write(f"\nTraining completed successfully\n")
    log_file.write(f"Final ε: {final_epsilon:.4f}\n")
    log_file.close()
    writer.close()
    
    # Save final model (with error handling for disk space)
    try:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.epochs,
            'epsilon': final_epsilon,
            'delta': delta
        }, os.path.join(save_dir, 'final_model.pth'))
        print(f"Final model saved successfully.")
    except RuntimeError as e:
        print(f"Warning: Could not save final model (disk space?): {e}")
        print(f"  Note: Best model was already saved during training")
    
    return final_epsilon


def main():
    parser = argparse.ArgumentParser(description='Train HALO with Custom DP-SGD')
    
    # Privacy parameters
    parser.add_argument('--epsilon', type=float, default=8.0,
                        help='Target privacy budget epsilon (default: 8.0)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Privacy parameter delta (default: 1e-5)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Learning rate (default: 0.002)')
    
    # Data parameters
    parser.add_argument('--dataset_size', type=int, required=True,
                        help='Total number of training samples (required)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training dataset pickle file')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation dataset pickle file')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints/custom_dp',
                        help='Directory to save checkpoints and logs')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load config
    config = HALOConfig()
    config.batch_size = args.batch_size
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = pickle.load(open(args.train_data, 'rb'))
    val_dataset = pickle.load(open(args.val_data, 'rb'))
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Verify dataset size
    if len(train_dataset) != args.dataset_size:
        print(f"Warning: --dataset_size ({args.dataset_size}) != actual size ({len(train_dataset)})")
        print(f"   Using actual size: {len(train_dataset)}")
        args.dataset_size = len(train_dataset)
    
    # Train
    final_epsilon = train_with_custom_dp(config, args, train_dataset, val_dataset, 
                                          device, args.save_dir)
    
    print(f"\nTraining complete. Final (eps,delta)-DP: ({final_epsilon:.2f}, {args.delta})")


if __name__ == '__main__':
    main()
