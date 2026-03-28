#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from src.arc_dataset import ARCTaskDataset
from src.arc_dataloader import get_arc_loader
from src.model import ARCFewShotHRM  # Structurally aligned import

# =========================
# Visual Artifact Dependency
# =========================
ARC_HEX_PALETTE = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
ARC_CMAP = colors.ListedColormap(ARC_HEX_PALETTE)
ARC_NORM = colors.Normalize(vmin=0, vmax=9)

def preserve_visual_state(qx, qy, pred, task_id, save_dir="eval_visual_artifacts"):
    """
    Reconstructs the spatial mapping of the task and preserves it as a visual artifact.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Isolate spatial grids by stripping batch/channel dimensions
    input_grid = qx.squeeze().cpu().numpy()
    truth_grid = qy.squeeze().cpu().numpy()
    pred_grid = pred.argmax(dim=1).squeeze().cpu().numpy()

    # Handle structural edge cases where multiple query examples exist
    if input_grid.ndim > 2: input_grid = input_grid[-1]
    if truth_grid.ndim > 2: truth_grid = truth_grid[-1]
    if pred_grid.ndim > 2: pred_grid = pred_grid[-1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[0].set_title("Input (Query X)", fontsize=10, pad=10)
    axes[0].axis('off')

    axes[1].imshow(truth_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[1].set_title("Ground Truth (Query Y)", fontsize=10, pad=10)
    axes[1].axis('off')

    axes[2].imshow(pred_grid, cmap=ARC_CMAP, norm=ARC_NORM)
    axes[2].set_title("Model Output (Prediction)", fontsize=10, pad=10)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"task_reconstruction_{task_id:04d}.png"), dpi=150)
    plt.close(fig) # Prevent memory leakage in the primary execution loop

# =========================
# Metrics
# =========================
def pixel_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    target = target.squeeze(1)
    return (pred == target).float().mean().item()

def task_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    target = target.squeeze(1)
    return (pred == target).all(dim=(1, 2)).float().mean().item()

# =========================
# SAFE TTA (Architecturally Aligned)
# =========================
def safe_tta(model, sx, sy, qx, steps, lr):
    """
    Reconstructs the precise TTA causal chain from training.
    Optimizes across all segments to map structural relationships.
    """
    local_model = copy.deepcopy(model)
    local_model.train()
    
    opt = torch.optim.Adam(local_model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        # Engage deep supervision causal mechanism
        segment_logits = local_model(sx, sy, sx, return_all_segments=True)
        loss = sum(F.cross_entropy(seg, sy.squeeze(1).long()) for seg in segment_logits) / len(segment_logits)
        loss.backward()
        opt.step()

    local_model.eval()
    with torch.no_grad():
        # Final inference step returns terminal segment logits
        return local_model(sx, sy, qx)

# =========================
# Main Execution Sequence
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    # Architectural instantiation arguments
    parser.add_argument("--t_steps", type=int, default=4)
    parser.add_argument("--max_segments", type=int, default=8)
    parser.add_argument("--tta_steps", type=int, default=10)
    parser.add_argument("--tta_lr", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"eval_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Execution Environment: {device}")

    dataset = ARCTaskDataset(args.data_dir)
    loader = get_arc_loader(dataset, batch_size=args.batch_size)

    # Architectural Instantiation
    model = ARCFewShotHRM(dim=128, T_steps=args.t_steps, max_segments=args.max_segments).to(device)

    # State Extraction Logic
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) 
    model.load_state_dict(state_dict)
    model.eval()

    total_pixel_acc = 0
    total_task_acc = 0
    count = 0

    logging.info("Initiating structural evaluation sequence...")

    for batch in tqdm(loader):
        if batch["query_y"] is None:
            continue

        support_x = batch["support_x"].to(device)
        support_y = batch["support_y"].to(device)
        query_x = batch["query_x"].to(device)
        query_y = batch["query_y"].to(device)

        support_mask = batch["support_mask"]
        query_mask = batch["query_mask"]

        B = support_x.shape[0]

        for b in range(B):
            # SAFE slicing mechanism
            s_len = int(support_mask[b].sum().item())
            q_len = int(query_mask[b].sum().item())

            sx = support_x[b, :s_len].long()
            sy = support_y[b, :s_len].long()
            qx = query_x[b, :q_len].long()
            qy = query_y[b, :q_len].long()

            if sx.numel() == 0 or qx.numel() == 0:
                continue

            # =========================
            # Execute Aligned TTA
            # =========================
            pred = safe_tta(model, sx, sy, qx, steps=args.tta_steps, lr=args.tta_lr)

            # =========================
            # Dimensional Calibration
            # =========================
            h, w = qy.shape[-2:]
            pred = pred[:, :, :h, :w]

            # =========================
            # Metric Aggregation & Visualization
            # =========================
            p_acc = pixel_accuracy(pred, qy)
            t_acc = task_accuracy(pred, qy)

            # Isolate and preserve the visual state matrix
            preserve_visual_state(qx, qy, pred, count)

            total_pixel_acc += p_acc
            total_task_acc += t_acc
            count += 1

            logging.info(f"Evaluation Vector {count}: Pixel Acc = {p_acc:.4f}, Task Acc = {t_acc:.4f}")

    if count > 0:
        logging.info("=" * 50)
        logging.info(f"Aggregate Pixel Accuracy: {total_pixel_acc / count:.4f}")
        logging.info(f"Aggregate Task Accuracy:  {total_task_acc / count:.4f}")
        logging.info("=" * 50)
    else:
        logging.info("Insufficient information: No valid vectors evaluated.")