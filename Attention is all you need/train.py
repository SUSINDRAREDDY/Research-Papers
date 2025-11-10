import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import train_dataset, valid_dataset, src_vocab, tgt_vocab, create_batches
from transformer import Transformer, make_src_mask, make_tgt_mask
import math

SRC_PAD_IDX = src_vocab["<pad>"]
TGT_PAD_IDX = tgt_vocab["<pad>"]

SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE = len(tgt_vocab)

D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 256
DROPOUT = 0.1

BATCH_SIZE = 128      
EPOCHS = 10
LR = 1e-4 
WARMUP_STEPS = 200
CLIP_NORM = 1.0 

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Training on:", device)

model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                    d_model=D_MODEL, num_layers=NUM_LAYERS,
                    num_heads=NUM_HEADS, hidden_layers=D_FF, dropout=DROPOUT).to(device)

def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)

criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

global_step = 0

def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step / WARMUP_STEPS)
    else:
        return LR * (WARMUP_STEPS ** 0.5) * (step ** -0.5)

def pad_batch(batch_src, batch_tgt):
    batch_src = [s if len(s) > 0 else [SRC_PAD_IDX] for s in batch_src]
    batch_tgt = [t if len(t) > 0 else [TGT_PAD_IDX] for t in batch_tgt]

    max_src = max(len(s) for s in batch_src)
    max_tgt = max(len(t) for t in batch_tgt)
    src_pad = [s + [SRC_PAD_IDX] * (max_src - len(s)) for s in batch_src]
    tgt_pad = [t + [TGT_PAD_IDX] * (max_tgt - len(t)) for t in batch_tgt]

    src_tensor = torch.tensor(src_pad, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(tgt_pad, dtype=torch.long, device=device)
    return src_tensor, tgt_tensor

def train_one_epoch(epoch):
    global global_step
    model.train()
    total_loss = 0
    batches = create_batches(train_dataset, batch_size=BATCH_SIZE)

    for i, (src_batch, tgt_batch) in enumerate(batches):
        global_step += 1
        
        # Update learning rate with warmup
        current_lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        src_tensor, tgt_tensor = pad_batch(src_batch, tgt_batch)
        tgt_input = tgt_tensor[:, :-1]     # decoder input
        tgt_output = tgt_tensor[:, 1:]     # expected output
        src_mask = make_src_mask(src_tensor, SRC_PAD_IDX).to(device)
        tgt_mask = make_tgt_mask(tgt_input, TGT_PAD_IDX).to(device)
        logits = model(src_tensor, tgt_input, src_mask, tgt_mask)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), tgt_output.reshape(B*T))
        optimizer.zero_grad()
        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        
        # Update weights
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Epoch {epoch}, Step {i+1}/{len(batches)}, Loss: {loss.item():.4f}, "
                  f"GradNorm: {total_norm:.4f}, LR: {current_lr:.6f}")

    avg_loss = total_loss / len(batches)
    print(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}\n")
    return True

for epoch in range(1, EPOCHS + 1):
    success = train_one_epoch(epoch)
    if not success:
        print("Training stopped due to numerical instability")
        break

print("Training complete:)")

# Save model
torch.save(model.state_dict(), "transformer_safe.pth")
print("Model saved to transformer_safe.pth")