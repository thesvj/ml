import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

from src.dataloader import get_dataloaders_standard, get_dataloaders_meta
from src.model import MetaOHLCVPredictor


# =========================
# Args
# =========================
parser = argparse.ArgumentParser(description="Evaluate trained meta-learning model")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--meta", action="store_true", help="Use meta-learning evaluation")
parser.add_argument("--lookback", type=int, default=10, help="Sequence length")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--k_shot", type=int, default=5, help="Support samples per task")
parser.add_argument("--q_query", type=int, default=1, help="Query samples per task")
args = parser.parse_args()


# =========================
# Setup
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info(f"Device: {device}")


# =========================
# Load data and model
# =========================
if not Path(args.checkpoint).exists():
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

checkpoint = torch.load(args.checkpoint, map_location=device)
logging.info(f"Loading checkpoint: {args.checkpoint}")

model_args = checkpoint.get("args", {})

# Load test dataset
if args.meta:
    _, _, test_loader, _, _, test_dataset = get_dataloaders_meta(
        batch_size=args.batch_size,
        lookback=args.lookback,
        k_shot=args.k_shot,
        q_query=args.q_query
    )
else:
    _, _, test_loader, _, _, test_dataset = get_dataloaders_standard(
        batch_size=args.batch_size,
        lookback=args.lookback
    )

# Get input/output sizes
sample_batch = next(iter(test_loader))
if args.meta:
    input_size = sample_batch['support_x'].shape[-1]
    output_size = sample_batch['support_y'].shape[-1]
else:
    input_size = sample_batch['x'].shape[-1]
    output_size = sample_batch['y'].shape[-1]

# Create model
model = MetaOHLCVPredictor(
    input_size=input_size,
    dim=model_args.get("hidden_size", 128),
    num_heads=model_args.get("num_heads", 4),
    output_size=output_size
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

logging.info(f"Test samples: {len(test_dataset)}")
logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# =========================
# Evaluation
# =========================
criterion = nn.MSELoss()
total_loss = 0
predictions = []
targets = []

logging.info("Evaluating on test set...")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if args.meta:
            support_x = batch['support_x'].to(device)
            support_y = batch['support_y'].to(device)
            query_x = batch['query_x'].to(device)
            query_y = batch['query_y'].to(device)
            
            pred = model(support_x, support_y, query_x)
        else:
            x = batch['x'].to(device)
            query_y = batch['y'].to(device)
            
            pred = model.forward_standard(x)
        
        loss = criterion(pred, query_y)
        total_loss += loss.item()
        
        predictions.append(pred.cpu())
        targets.append(query_y.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Batch {batch_idx+1} | Loss: {loss.item():.6f}")

# Aggregate results
avg_loss = total_loss / len(test_loader)
predictions = torch.cat(predictions, dim=0)
targets = torch.cat(targets, dim=0)

# Calculate metrics
mse = criterion(predictions, targets).item()
mae = torch.abs(predictions - targets).mean().item()
rmse = torch.sqrt(torch.tensor(mse)).item()

# R²
ss_res = ((predictions - targets) ** 2).sum().item()
ss_tot = ((targets - targets.mean()) ** 2).sum().item()
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

logging.info("=" * 80)
logging.info(f"Test Results:")
logging.info(f"  MSE:  {mse:.6f}")
logging.info(f"  RMSE: {rmse:.6f}")
logging.info(f"  MAE:  {mae:.6f}")
logging.info(f"  R²:   {r2:.6f}")
logging.info("=" * 80)

# Save results
results = {
    "checkpoint": str(args.checkpoint),
    "mode": "meta-learning" if args.meta else "standard",
    "lookback": args.lookback,
    "num_test_samples": len(test_dataset),
    "metrics": {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
}

results_file = f"logs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

logging.info(f"Results saved to {results_file}")
