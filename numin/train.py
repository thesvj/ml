import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
from datetime import datetime
import os
import argparse
import json
from pathlib import Path

from src.dataloader import get_dataloaders_standard, get_dataloaders_meta
from src.model import MetaOHLCVPredictor


# =========================
# Args
# =========================
parser = argparse.ArgumentParser(description="Train Meta-Learning OHLCV-to-Returns predictor")
parser.add_argument("--meta", action="store_true", help="Use meta-learning training")
parser.add_argument("--lookback", type=int, default=10, help="Sequence length")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size (tasks for meta-learning)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--hidden_size", type=int, default=128, help="Hidden dimension")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--k_shot", type=int, default=5, help="Support samples per task (meta-learning)")
parser.add_argument("--q_query", type=int, default=1, help="Query samples per task (meta-learning)")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()


# =========================
# Setup
# =========================
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Setup logging
log_file = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

mode = "meta-learning" if args.meta else "standard"
logging.info(f"Mode: {mode}")
logging.info(f"Device: {device}")
logging.info(f"Arguments: {vars(args)}")


# =========================
# Data
# =========================
if args.meta:
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        get_dataloaders_meta(
            batch_size=args.batch_size,
            lookback=args.lookback,
            num_workers=2,
            k_shot=args.k_shot,
            q_query=args.q_query
        )
else:
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        get_dataloaders_standard(
            batch_size=args.batch_size,
            lookback=args.lookback,
            num_workers=2
        )

logging.info(f"Train samples: {len(train_dataset)}")
logging.info(f"Val samples: {len(val_dataset)}")
logging.info(f"Test samples: {len(test_dataset)}")

# Get input/output sizes
sample_batch = next(iter(train_loader))
if args.meta:
    input_size = sample_batch['support_x'].shape[-1]
    output_size = sample_batch['support_y'].shape[-1]
else:
    input_size = sample_batch['x'].shape[-1]
    output_size = sample_batch['y'].shape[-1]

logging.info(f"Input size: {input_size}, Output size: {output_size}")


# =========================
# Model
# =========================
model = MetaOHLCVPredictor(
    input_size=input_size,
    dim=args.hidden_size,
    num_heads=args.num_heads,
    output_size=output_size
)

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Total parameters: {total_params:,}")
logging.info(f"Trainable parameters: {trainable_params:,}")

for name, module in model.named_children():
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logging.info(f"  {name:15s} | total: {total:,} | trainable: {trainable:,}")


# =========================
# Optimizer + Scheduler
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True,
    min_lr=1e-6
)

criterion = nn.MSELoss()


# =========================
# Resume
# =========================
start_epoch = 0
best_val_loss = float("inf")

if args.resume and os.path.exists(args.resume):
    logging.info(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    
    logging.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.6f}")


# =========================
# Training Functions
# =========================
def train_epoch_meta(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training (Meta)")
    for batch in pbar:
        support_x = batch['support_x'].to(device)
        support_y = batch['support_y'].to(device)
        query_x = batch['query_x'].to(device)
        query_y = batch['query_y'].to(device)
        
        optimizer.zero_grad()
        
        pred = model(support_x, support_y, query_x)
        loss = criterion(pred, query_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_loader)


def train_epoch_standard(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training (Standard)")
    for batch in pbar:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        
        pred = model.forward_standard(x)
        loss = criterion(pred, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_loader)


def evaluate_meta(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating (Meta)")
        for batch in pbar:
            support_x = batch['support_x'].to(device)
            support_y = batch['support_y'].to(device)
            query_x = batch['query_x'].to(device)
            query_y = batch['query_y'].to(device)
            
            pred = model(support_x, support_y, query_x)
            loss = criterion(pred, query_y)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(eval_loader)


def evaluate_standard(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating (Standard)")
        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            pred = model.forward_standard(x)
            loss = criterion(pred, y)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(eval_loader)


# =========================
# Training Loop
# =========================
logging.info("=" * 80)
logging.info(f"Starting training ({mode} mode)...")
logging.info("=" * 80)

train_fn = train_epoch_meta if args.meta else train_epoch_standard
eval_fn = evaluate_meta if args.meta else evaluate_standard
mode_suffix = "meta" if args.meta else "standard"

for epoch in range(start_epoch, args.epochs):
    train_loss = train_fn(model, train_loader, optimizer, criterion, device)
    val_loss = eval_fn(model, val_loader, criterion, device)
    
    logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = f"models/meta_ohlcv_{mode_suffix}_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args)
        }, checkpoint_path)
        logging.info(f"✓ Saved best model to {checkpoint_path}")
    
    # Save periodic checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"models/meta_ohlcv_{mode_suffix}_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args)
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")


# =========================
# Test Evaluation
# =========================
logging.info("=" * 80)
logging.info("Evaluating on test set...")
logging.info("=" * 80)

best_model_path = f"models/meta_ohlcv_{mode_suffix}_best.pt"
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss = eval_fn(model, test_loader, criterion, device)
logging.info(f"Test Loss: {test_loss:.6f}")

# Save metrics
metrics = {
    "mode": mode,
    "best_val_loss": best_val_loss,
    "test_loss": test_loss,
    "args": vars(args)
}

with open(f"logs/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    json.dump(metrics, f, indent=2)

logging.info("Training complete!")
