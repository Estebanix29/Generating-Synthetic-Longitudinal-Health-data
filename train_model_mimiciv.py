import os
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import HALOModel
from config_mimiciv import HALOConfig
from datetime import datetime
import time

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# PyTorch Performance Optimizations
torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
torch.set_float32_matmul_precision('high')  # Use TensorFloat-32 on compatible hardware

config = HALOConfig()

local_rank = -1
fp16 = False
if local_rank == -1:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

train_ehr_dataset = pickle.load(open('./data/trainDataset.pkl', 'rb'))
val_ehr_dataset = pickle.load(open('./data/valDataset.pkl', 'rb'))

def get_batch(loc, batch_size, mode):
  # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
  #   Where each patient P is [V_1, V_2, ... , V_j]
  #     Where each visit V is [C_1, C_2, ... , C_k]
  #   And where each Label L is a binary vector [L_1 ... L_n]
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
      if j+2 >= config.n_ctx:  # Skip visits that exceed context length
        break
      batch_ehr[i,j+2][v] = 1
      batch_mask[i,j+2] = 1
    batch_ehr[i,1,config.code_vocab_size:config.code_vocab_size+config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
    visit_end_idx = min(len(visits)+1, config.n_ctx-1)  # Ensure context length is not exceeded
    batch_ehr[i,visit_end_idx,config.code_vocab_size+config.label_vocab_size+1] = 1 # Set the final visit to have the end token
    if visit_end_idx + 1 < config.n_ctx:
      batch_ehr[i,visit_end_idx+1:,config.code_vocab_size+config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_mask[:,1] = 1 # Set the mask to cover the labels
  batch_ehr[:,0,config.code_vocab_size+config.label_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_mask

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

print("\n" + "-"*60)
print("Initializing model")
print("-"*60)
print(f"Creating model with {config.total_vocab_size} vocabulary size...")
print(f"This may take 2-5 minutes for large autoregressive mask creation...")
print("-"*60 + "\n")

model = HALOModel(config).to(device)

print("\nModel initialized successfully.")
print(f"Total parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

# PyTorch 2.0+ optimization: Compile model for faster execution
if hasattr(torch, 'compile'):
  print("Compiling model with torch.compile for better performance...")
  model = torch.compile(model)

# Use fused Adam optimizer for better performance
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, fused=False)  # fused=True requires CUDA

if os.path.exists("./save/halo_model_mimiciv"):
  print("Loading previous model")
  checkpoint = torch.load('./save/halo_model_mimiciv', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# Train
global_loss = 1e10

print("\n" + "-"*60)
print("Training started")
print("-"*60)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device}")
print(f"Total Epochs: {config.epoch}")
print(f"Training Samples: {len(train_ehr_dataset):,}")
print(f"Validation Samples: {len(val_ehr_dataset):,}")
print(f"Batch Size: {config.batch_size}")
print(f"Batches per Epoch: {len(train_ehr_dataset) // config.batch_size:,}")
print(f"Learning Rate: {config.lr}")
print("-"*60 + "\n")

training_start_time = time.time()

for e in range(config.epoch):
  epoch_start_time = time.time()
  print(f"\n{'='*80}")
  print(f"EPOCH {e+1}/{config.epoch} - Started at {datetime.now().strftime('%H:%M:%S')}")
  print(f"{'='*80}")
  
  shuffle_training_data(train_ehr_dataset)
  epoch_loss = 0.0
  num_batches = 0
  
  for i in range(0, len(train_ehr_dataset), config.batch_size):
    model.train()
    
    batch_ehr, batch_mask = get_batch(i, config.batch_size, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
    # Use set_to_none for better performance than zero_grad()
    optimizer.zero_grad(set_to_none=True)
    
    loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
    loss.backward()
    optimizer.step()
    
    epoch_loss += loss.item()
    num_batches += 1
    
    # Log every 500 batches (or ~8000 samples)
    if i % (500*config.batch_size) == 0 and i > 0:
      elapsed = time.time() - epoch_start_time
      progress_pct = (i / len(train_ehr_dataset)) * 100
      samples_processed = min(i + config.batch_size, len(train_ehr_dataset))
      avg_loss_so_far = epoch_loss / num_batches
      
      print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
            f"Batch {num_batches:,}/{len(train_ehr_dataset) // config.batch_size:,} "
            f"({progress_pct:.1f}%) | "
            f"Samples: {samples_processed:,}/{len(train_ehr_dataset):,} | "
            f"Loss: {loss.item() * 8:.6f} | "
            f"Avg Loss: {avg_loss_so_far * 8:.6f} | "
            f"Time: {elapsed/60:.1f}m")
    if i % (500*config.batch_size) == 0:
      if i == 0:
        continue
    
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), config.batch_size):
          batch_ehr, batch_mask = get_batch(v_i, config.batch_size, 'valid')
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
  
          val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=config.pos_loss_weight)
          val_l.append((val_loss).cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        print(f"\n  >>> Validation Loss: {cur_val_loss:.7f} (Best: {global_loss:.7f})")
        
        if cur_val_loss < global_loss:
          global_loss = cur_val_loss
          state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': i
            }
          torch.save(state, './save/halo_model_mimiciv')
          print(f"  New best model saved. (Loss improved from {global_loss:.7f} to {cur_val_loss:.7f})\n")
  
  # Print epoch summary
  epoch_time = time.time() - epoch_start_time
  avg_epoch_loss = epoch_loss / num_batches
  total_elapsed = time.time() - training_start_time
  
  print(f"\n{'-'*80}")
  print(f"EPOCH {e+1}/{config.epoch} COMPLETE - {datetime.now().strftime('%H:%M:%S')}")
  print(f"{'-'*80}")
  print(f"  Average Training Loss: {avg_epoch_loss * 8:.6f}")
  print(f"  Batches Processed: {num_batches:,}")
  print(f"  Epoch Duration: {epoch_time/60:.1f} minutes ({epoch_time/3600:.2f} hours)")
  print(f"  Total Training Time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
  print(f"  Best Validation Loss So Far: {global_loss:.7f}")
  
  # Estimate remaining time
  avg_epoch_time = total_elapsed / (e + 1)
  remaining_epochs = config.epoch - (e + 1)
  estimated_remaining = avg_epoch_time * remaining_epochs
  
  if remaining_epochs > 0:
    print(f"  Estimated Time Remaining: {estimated_remaining/60:.1f} minutes ({estimated_remaining/3600:.2f} hours, {estimated_remaining/3600/24:.2f} days)")
    eta = datetime.now().timestamp() + estimated_remaining
    eta_str = datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  Estimated Completion: {eta_str}")
  
  print(f"{'-'*80}\n")

# Final summary
print("\n" + "-"*60)
print("Training complete")
print("-"*60)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Duration: {(time.time() - training_start_time)/3600:.2f} hours ({(time.time() - training_start_time)/3600/24:.2f} days)")
print(f"Best Validation Loss: {global_loss:.7f}")
print(f"Model saved to: ./save/halo_model_mimiciv")
print("-"*60 + "\n")
