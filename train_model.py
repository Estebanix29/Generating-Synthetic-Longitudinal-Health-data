'''
    Training script for the HALO model architecture (from the paper)
    
    For MIMIC-III:
    - Run: python train_model.py
    - Memory: ~8-12GB GPU
    - Time: ~4-8 hours for 50 epochs
'''
import os
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import HALOModel
from config import HALOConfig

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = HALOConfig()

# Model save path
SAVE_PATH = './save/halo_model'

local_rank = -1
fp16 = False
if local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {device}")
print(f"Number of GPUs: {n_gpu}")
print(f"Config: n_embd={config.n_embd}, total_vocab={config.total_vocab_size}, code_vocab={config.code_vocab_size}")
print(f"FineAutoregressiveHead layer size: {config.n_embd + config.total_vocab_size}")

# Load data
print("Loading datasets...")
train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))
print(f"Train: {len(train_ehr_dataset)} patients, Val: {len(val_ehr_dataset)} patients")

def get_batch(loc, batch_size, mode):
    """
    EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
      Where each patient P is [V_1, V_2, ... , V_j]
        Where each visit V is [C_1, C_2, ... , C_k]
      And where each Label L is a binary vector [L_1 ... L_n]
    """
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc+batch_size]
    else:
        ehr = test_ehr_dataset[loc:loc+batch_size]
    
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.total_vocab_size))
    batch_mask = np.zeros((len(ehr), config.n_ctx, 1))
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_ehr[i,j+2][v] = 1
            batch_mask[i,j+2] = 1
        batch_ehr[i,1,config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels'])
        batch_ehr[i,len(visits)+1,config.code_vocab_size+config.label_vocab_size+1] = 1
        batch_ehr[i,len(visits)+2:,config.code_vocab_size+config.label_vocab_size+2] = 1
    
    batch_mask[:,1] = 1
    batch_ehr[:,0,config.code_vocab_size+config.label_vocab_size] = 1
    batch_mask = batch_mask[:,1:,:]
    return batch_ehr, batch_mask

def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)

# Initialize model
print("Initializing ORIGINAL HALO model...")
model = HALOModel(config).to(device)

# Print model size
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# Check for existing checkpoint
if os.path.exists(SAVE_PATH):
    print(f"Loading previous model from {SAVE_PATH}")
    checkpoint = torch.load(SAVE_PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0)
    global_loss = checkpoint.get('val_loss', 1e10)
    print(f"Resuming from epoch {start_epoch}, best val_loss: {global_loss:.6f}")
else:
    start_epoch = 0
    global_loss = 1e10
    print("Starting fresh training")

# Create save directory
os.makedirs('./save', exist_ok=True)

# Train
print(f"\nStarting training for {config.epoch} epochs...")
for e in tqdm(range(start_epoch, config.epoch), desc="Epochs"):
    shuffle_training_data(train_ehr_dataset)
    epoch_losses = []
    
    for i in range(0, len(train_ehr_dataset), config.batch_size):
        model.train()
        
        batch_ehr, batch_mask = get_batch(i, config.batch_size, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        
        if i % (500*config.batch_size) == 0:
            print(f"Epoch {e}, Iter {i}: Training Loss: {loss.item():.6f}")
        
        # Validation every 500 batches
        if i % (500*config.batch_size) == 0 and i > 0:
            model.eval()
            with torch.no_grad():
                val_l = []
                for v_i in range(0, len(val_ehr_dataset), config.batch_size):
                    batch_ehr, batch_mask = get_batch(v_i, config.batch_size, 'valid')
                    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
                    
                    val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
                    val_l.append(val_loss.cpu().detach().numpy())
                
                cur_val_loss = np.mean(val_l)
                print(f"Epoch {e} Validation Loss: {cur_val_loss:.7f}")
                
                if cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': e,
                        'iteration': i,
                        'val_loss': cur_val_loss
                    }
                    torch.save(state, SAVE_PATH)
                    print('\n------------ Save best model ------------\n')
    
    # End of epoch summary
    avg_epoch_loss = np.mean(epoch_losses)
    print(f"Epoch {e} complete. Avg training loss: {avg_epoch_loss:.6f}")

print(f"\nTraining complete! Best validation loss: {global_loss:.6f}")
print(f"Model saved to: {SAVE_PATH}")
