import argparse
import importlib.util
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items if items else None


def _load_yaml_config(path: str) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required for --config support. Install with: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got: {type(config)}")
    return config


def _set_arg_if_present(args: argparse.Namespace, key: str, value) -> None:
    if value is not None and hasattr(args, key):
        setattr(args, key, value)


def _apply_inference_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    cfg = _load_yaml_config(args.config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    infer_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference", {}), dict) else {}

    _set_arg_if_present(args, "data_dir", data_cfg.get("augmented_dataset_path"))
    _set_arg_if_present(args, "task_types", data_cfg.get("task_types"))
    _set_arg_if_present(args, "task_ids", data_cfg.get("task_ids"))
    _set_arg_if_present(args, "num_workers", data_cfg.get("num_workers"))
    pin_memory = data_cfg.get("pin_memory")
    if pin_memory is not None:
        args.no_pin_memory = not bool(pin_memory)

    _set_arg_if_present(args, "max_size", model_cfg.get("max_size"))
    _set_arg_if_present(args, "num_timesteps", model_cfg.get("num_timesteps"))

    _set_arg_if_present(args, "checkpoint", infer_cfg.get("checkpoint"))
    _set_arg_if_present(args, "batch_size", infer_cfg.get("batch_size"))
    _set_arg_if_present(args, "output_dir", infer_cfg.get("output_dir"))
    if not args.checkpoint:
        raise ValueError("Checkpoint is required. Set --checkpoint or inference.checkpoint in --config")
    return args


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("arc_diff_inference")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _model_param_stats(model: torch.nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def _load_local_module(module_filename: str, module_name: str):
    module_path = CURRENT_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_filename} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_dataloader_module = _load_local_module("dataloader.py", "arc_diff_dataloader")
_model_module = _load_local_module("model.py", "arc_diff_model")

get_test_dataloader = _dataloader_module.get_test_dataloader
ARCDiffusionModel = _model_module.ARCDiffusionModel


class DiscreteNoiseScheduler:
    def __init__(self, num_timesteps: int = 50, vocab_size: int = 10):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


def pad_batch_to_model_size(batch: Dict[str, torch.Tensor], model_size: int, pad_value: int = 10):
    input_grid = batch["input_grid"]
    output_grid = batch["output_grid"]
    input_mask = batch["input_mask"]
    output_mask = batch["output_mask"]

    current_height, current_width = input_grid.shape[-2:]
    if current_height > model_size or current_width > model_size:
        raise ValueError(
            f"Batch grid size {current_height}x{current_width} exceeds model size {model_size}x{model_size}."
        )

    def pad_tensor(tensor: torch.Tensor, fill_value):
        padded = torch.full(
            (tensor.shape[0], model_size, model_size),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, :tensor.shape[-2], :tensor.shape[-1]] = tensor
        return padded

    return {
        "input_grid": pad_tensor(input_grid, pad_value),
        "output_grid": pad_tensor(output_grid, pad_value),
        "input_mask": pad_tensor(input_mask, False),
        "output_mask": pad_tensor(output_mask, False),
        "task_idx": batch["task_idx"],
        "height": batch["height"],
        "width": batch["width"],
        "d4_idx": batch["d4_idx"],
        "color_shift": batch["color_shift"],
        "task_ids": batch["task_ids"],
        "file_name": batch["file_name"],
        "has_output": batch["has_output"],
    }


def _crop_grid(grid: torch.Tensor, height: int, width: int):
    return grid[:height, :width].detach().cpu().tolist()


def _infer_input_size(mask_2d: torch.Tensor) -> Tuple[int, int]:
    rows = int(mask_2d.any(dim=1).sum().item())
    cols = int(mask_2d.any(dim=0).sum().item())
    return rows, cols


def _new_group_stats():
    return {
        "total_examples": 0,
        "labeled_examples": 0,
        "cell_correct": 0.0,
        "cell_total": 0.0,
        "task_correct": 0,
        "task_total": 0,
    }


def _update_group_stats(stats: Dict[str, float], has_label: bool, correct_cells: float, total_cells: float, exact_match: bool):
    stats["total_examples"] += 1
    if not has_label:
        return

    stats["labeled_examples"] += 1
    stats["cell_correct"] += float(correct_cells)
    stats["cell_total"] += float(total_cells)
    stats["task_total"] += 1
    if exact_match:
        stats["task_correct"] += 1


def _finalize_group_stats(stats_map: Dict[str, Dict[str, float]]):
    finalized = {}
    for key in sorted(stats_map.keys()):
        item = stats_map[key]
        cell_total = item["cell_total"]
        task_total = item["task_total"]
        finalized[key] = {
            "total_examples": item["total_examples"],
            "labeled_examples": item["labeled_examples"],
            "unlabeled_examples": item["total_examples"] - item["labeled_examples"],
            "cell_accuracy": (item["cell_correct"] / cell_total) if cell_total > 0 else None,
            "task_accuracy": (item["task_correct"] / task_total) if task_total > 0 else None,
            "cell_correct": item["cell_correct"],
            "cell_total": cell_total,
            "task_correct": item["task_correct"],
            "task_total": task_total,
        }
    return finalized


@torch.no_grad()
def run_inference(args):
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{Path(args.checkpoint).stem}_{run_tag}"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    logger = _setup_logger(log_path)

    tracker_path = Path(args.tracker_file)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Starting inference run %s on %s", run_name, device)
    logger.info("Logging to %s", log_path)

    loader = get_test_dataloader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        max_size=args.max_size,
        task_types=args.task_types,
        task_ids=args.task_ids,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        pad_value=args.pad_value,
    )

    model = ARCDiffusionModel(max_size=args.max_size).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    param_stats = _model_param_stats(model)
    checkpoint_size_bytes = Path(args.checkpoint).stat().st_size

    scheduler = DiscreteNoiseScheduler(num_timesteps=args.num_timesteps).to(device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    _append_jsonl(
        tracker_path,
        {
            "mode": "inference",
            "event": "run_started",
            "run_tag": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "checkpoint": str(args.checkpoint),
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "output_dir": str(output_root),
            "data_dir": args.data_dir,
            "batch_size": args.batch_size,
            **param_stats,
        },
    )

    total_examples = 0
    labeled_examples = 0
    cell_correct = 0.0
    cell_total = 0.0
    task_correct = 0
    task_total = 0
    per_file_counter = defaultdict(int)
    pattern_metrics = defaultdict(_new_group_stats)
    task_metrics = defaultdict(_new_group_stats)

    run_status = "ok"
    error_message = None

    try:
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            batch = pad_batch_to_model_size(batch, model.max_size, pad_value=args.pad_value)

            input_grid = batch["input_grid"].to(device)
            output_grid = batch["output_grid"].to(device)
            input_mask = batch["input_mask"].to(device)
            output_mask = batch["output_mask"].to(device)
            task_ids = batch["task_idx"].to(device)
            has_output = batch["has_output"].to(device)

            batch_size = input_grid.shape[0]
            timesteps = torch.full((batch_size,), scheduler.num_timesteps - 1, dtype=torch.long, device=device)
            alpha_bar = scheduler.alpha_bars[timesteps].clamp(1e-5, 1 - 1e-5)
            logsnr = torch.log(alpha_bar) - torch.log(1 - alpha_bar)

            xt = torch.randint(0, scheduler.vocab_size, (batch_size, model.max_size, model.max_size), device=device)
            effective_mask = torch.where(has_output[:, None, None], output_mask, input_mask)

            logits_prev = model(
                xt=xt,
                input_grid=input_grid,
                task_ids=task_ids,
                logsnr=logsnr,
                d4_idx=batch["d4_idx"].to(device),
                color_shift=batch["color_shift"].to(device),
                masks=effective_mask,
                sc_p0=None,
            )
            sc = torch.log_softmax(logits_prev, dim=-1)
            sc = sc * effective_mask.unsqueeze(-1).float()

            logits = model(
                xt=xt,
                input_grid=input_grid,
                task_ids=task_ids,
                logsnr=logsnr,
                d4_idx=batch["d4_idx"].to(device),
                color_shift=batch["color_shift"].to(device),
                masks=effective_mask,
                sc_p0=sc,
            )
            predictions = logits.argmax(dim=-1)

            pred_h, pred_w = model.predict_sizes(
                input_grid=input_grid,
                task_ids=task_ids,
                d4_idx=batch["d4_idx"].to(device),
                color_shift=batch["color_shift"].to(device),
            )

            for i in range(batch_size):
                total_examples += 1
                task_name = str(batch["task_ids"][i])
                file_name = str(batch["file_name"][i])
                file_stem = Path(file_name).stem
                pattern_id = task_name
                task_id = f"{task_name}/{file_stem}"
                per_file_counter[(task_name, file_name)] += 1
                example_idx = per_file_counter[(task_name, file_name)] - 1

                in_h, in_w = _infer_input_size(input_mask[i])
                input_list = _crop_grid(input_grid[i], in_h, in_w)

                if bool(has_output[i].item()):
                    out_h = int(batch["height"][i].item())
                    out_w = int(batch["width"][i].item())
                else:
                    out_h = int(pred_h[i].item())
                    out_w = int(pred_w[i].item())

                out_h = max(1, min(out_h, model.max_size))
                out_w = max(1, min(out_w, model.max_size))

                pred_list = _crop_grid(predictions[i], out_h, out_w)

                groundtruth_list = None
                has_label = bool(has_output[i].item())
                correct_cells = 0.0
                total_cells = 0.0
                exact_match = False
                if has_label:
                    labeled_examples += 1
                    groundtruth_list = _crop_grid(output_grid[i], out_h, out_w)

                    mask_i = output_mask[i, :out_h, :out_w]
                    pred_i = predictions[i, :out_h, :out_w]
                    target_i = output_grid[i, :out_h, :out_w]

                    correct_cells = ((pred_i == target_i) & mask_i).float().sum().item()
                    total_cells = mask_i.float().sum().item()
                    cell_correct += float(correct_cells)
                    cell_total += float(total_cells)

                    task_total += 1
                    exact_match = bool(torch.equal(pred_i[mask_i], target_i[mask_i])) if mask_i.any() else True
                    if exact_match:
                        task_correct += 1

                _update_group_stats(pattern_metrics[pattern_id], has_label, correct_cells, total_cells, exact_match)
                _update_group_stats(task_metrics[task_id], has_label, correct_cells, total_cells, exact_match)

                task_dir = output_root / task_name
                task_dir.mkdir(parents=True, exist_ok=True)
                output_path = task_dir / f"{file_stem}_test_{example_idx:03d}.json"

                record = {
                    "task_id": task_name,
                    "file_name": file_name,
                    "example_index": example_idx,
                    "input": input_list,
                    "groundtruth": groundtruth_list,
                    "prediction": pred_list,
                }
                output_path.write_text(json.dumps(record), encoding="utf-8")
    except Exception as exc:
        run_status = "error"
        error_message = str(exc)
        logger.exception("Inference failed with error: %s", exc)
        _append_jsonl(
            tracker_path,
            {
                "mode": "inference",
                "event": "run_finished",
                "run_tag": run_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint": str(args.checkpoint),
                "checkpoint_size_bytes": checkpoint_size_bytes,
                "output_dir": str(output_root),
                "status": run_status,
                "error": error_message,
                "total_examples": total_examples,
                "labeled_examples": labeled_examples,
                "cell_accuracy": None,
                "task_accuracy": None,
                **param_stats,
            },
        )
        raise

    cell_acc = (cell_correct / cell_total) if cell_total > 0 else None
    task_acc = (task_correct / task_total) if task_total > 0 else None

    summary = {
        "total_examples": total_examples,
        "labeled_examples": labeled_examples,
        "unlabeled_examples": total_examples - labeled_examples,
        "cell_accuracy": cell_acc,
        "task_accuracy": task_acc,
        "cell_correct": cell_correct,
        "cell_total": cell_total,
        "task_correct": task_correct,
        "task_total": task_total,
    }

    summary_path = output_root / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pattern_summary = _finalize_group_stats(pattern_metrics)
    task_summary = _finalize_group_stats(task_metrics)
    pattern_summary_path = output_root / "metrics_pattern_wise.json"
    task_summary_path = output_root / "metrics_task_wise.json"
    pattern_summary_path.write_text(json.dumps(pattern_summary, indent=2), encoding="utf-8")
    task_summary_path.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")

    logger.info("Saved predictions to %s", output_root)
    logger.info("Saved metrics summary: %s", summary_path)
    logger.info("Saved pattern-wise metrics: %s", pattern_summary_path)
    logger.info("Saved task-wise metrics: %s", task_summary_path)
    logger.info("Summary: %s", json.dumps(summary))

    _append_jsonl(
        tracker_path,
        {
            "mode": "inference",
            "event": "run_finished",
            "run_tag": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": str(args.checkpoint),
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "output_dir": str(output_root),
            "status": run_status,
            "error": error_message,
            "total_examples": total_examples,
            "labeled_examples": labeled_examples,
            "cell_accuracy": cell_acc,
            "task_accuracy": task_acc,
            **param_stats,
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run ARC inference, save predictions, and compute metrics.")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", default="aug_data", help="Root folder with ARC task JSON files")
    parser.add_argument(
        "--task-types",
        type=_parse_csv_list,
        default=None,
        help="Comma-separated task type folders to include, e.g. 1d_flip,1d_fill",
    )
    parser.add_argument(
        "--task-ids",
        type=_parse_csv_list,
        default=None,
        help="Comma-separated task file stems to include, e.g. 1d_flip_0,1d_flip_1",
    )
    parser.add_argument("--output-dir", default="predictions", help="Directory where predictions will be saved")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-size", type=int, default=30, help="Maximum grid size supported by model")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pad-value", type=int, default=10, help="Pad token for ARC grids")
    parser.add_argument("--num-timesteps", type=int, default=50, help="Noise schedule steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for quick smoke tests")
    parser.add_argument("--log-dir", default="logs", help="Directory to save inference logs")
    parser.add_argument("--tracker-file", default="models/run_tracker.jsonl", help="JSONL file for run/model/error tracking")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args = _apply_inference_config(parsed_args)
    run_inference(parsed_args)