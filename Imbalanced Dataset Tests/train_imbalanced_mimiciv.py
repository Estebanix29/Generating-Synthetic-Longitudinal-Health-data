import os
import sys
import argparse
import pickle
import torch
import numpy as np
import random
from tqdm import tqdm
from model import HALOModel
from config import HALOConfig

def set_seed(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_batch(loc, batch_size, mode, train_dataset, val_dataset, config):
    """Same batching function as train_vast.py"""
    if mode == 'train':
        ehr = train_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_dataset[loc:loc+batch_size]
    else:
        raise ValueError("Invalid mode")
    
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

def train_model(scenario, output_suffix, data_dir=None):
    """Train HALO on imbalanced dataset"""
    
    print("-"*60)
    print(f"Training HALO on imbalanced dataset - scenario {scenario}")
    print("-"*60)
    print()
    
    # Set paths
    if data_dir is None:
        scenario_map = {
            1: 'data/imbalanced/scenario1_common_label13',
            2: 'data/imbalanced/scenario2_rare_label19',
            7: 'data/imbalanced/scenario7_90-10_label5'
        }
        data_dir = scenario_map.get(scenario)
    
    if data_dir is None:
        print(f"Error: Invalid scenario {scenario}")
        return
    
    train_path = f"{data_dir}/trainDataset.pkl"
    val_path = f"{data_dir}/valDataset.pkl"
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        print(f"Run: python imbalance_create_datasets.py --scenario {scenario}")
        return
    
    print(f"Loading data from: {data_dir}")
    
    # Load datasets
    with open(train_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_dataset = pickle.load(f)
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print()
    
    # Setup
    set_seed(4)
    config = HALOConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Initialize model
    model = HALOModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Training loop
    print("-"*60)
    print("Training start")
    print("-"*60)
    print()
    
    best_val_loss = float('inf')
    output_dir = f"save/halo_model_imbalanced_{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(config.epoch):
        model.train()
        train_loss = 0
        train_steps = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(train_dataset))
        shuffled_train = [train_dataset[i] for i in indices]
        
        progress_bar = tqdm(range(0, len(shuffled_train), config.batch_size), 
                           desc=f"Epoch {epoch+1}/{config.epoch}")
        
        for i in progress_bar:
            batch_size = min(config.batch_size, len(shuffled_train) - i)
            batch_ehr, batch_mask = get_batch(i, batch_size, 'train', shuffled_train, val_dataset, config)
            
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr,
                              ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.2f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for i in range(0, len(val_dataset), config.batch_size):
                batch_size = min(config.batch_size, len(val_dataset) - i)
                batch_ehr, batch_mask = get_batch(i, batch_size, 'valid', train_dataset, val_dataset, config)
                
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
                
                loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr,
                                  ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        print(f"Epoch {epoch+1}/{config.epoch}: Train Loss = {avg_train_loss:.2f}, Val Loss = {avg_val_loss:.2f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
            print(f"  Saved best model (val_loss={avg_val_loss:.2f})")
        
        print()
    
    print("-"*60)
    print("Training complete")
    print("-"*60)
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Model saved to: {output_dir}/pytorch_model.bin")
    print()
    print("Next step: Evaluate with evaluate_imbalanced_comparison.py")
    print()

def main():
    parser = argparse.ArgumentParser(description='Train HALO on imbalanced dataset')
    parser.add_argument('--scenario', type=int, required=True, choices=[1, 2, 7],
                        help='Scenario number (1, 2, or 7)')
    parser.add_argument('--output_suffix', type=str, required=True,
                        help='Suffix for output directory (e.g., "scenario1")')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Custom data directory (optional)')
    
    args = parser.parse_args()
    
    train_model(args.scenario, args.output_suffix, args.data_dir)

if __name__ == '__main__':
    main()
