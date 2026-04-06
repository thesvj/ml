import argparse
import importlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    tqdm = importlib.import_module("tqdm").tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

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


def _apply_train_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    cfg = _load_yaml_config(args.config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}

    _set_arg_if_present(args, "data_dir", data_cfg.get("augmented_dataset_path"))
    _set_arg_if_present(args, "task_types", data_cfg.get("task_types"))
    _set_arg_if_present(args, "task_ids", data_cfg.get("task_ids"))
    _set_arg_if_present(args, "num_workers", data_cfg.get("num_workers"))
    pin_memory = data_cfg.get("pin_memory")
    if pin_memory is not None:
        args.no_pin_memory = not bool(pin_memory)

    _set_arg_if_present(args, "max_size", model_cfg.get("max_size"))
    _set_arg_if_present(args, "num_timesteps", model_cfg.get("num_timesteps"))

    _set_arg_if_present(args, "epochs", train_cfg.get("epochs"))
    _set_arg_if_present(args, "batch_size", train_cfg.get("batch_size"))
    _set_arg_if_present(args, "eval_batch_size", train_cfg.get("eval_batch_size"))
    _set_arg_if_present(args, "lr", train_cfg.get("learning_rate"))
    _set_arg_if_present(args, "save_path", train_cfg.get("save_path"))
    _set_arg_if_present(args, "log_dir", train_cfg.get("log_dir"))
    _set_arg_if_present(args, "log_interval", train_cfg.get("log_interval"))
    return args


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("arc_diff_train")
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

get_dataloaders = _dataloader_module.get_dataloaders
ARCDiffusionModel = _model_module.ARCDiffusionModel


class DiscreteNoiseScheduler:
    def __init__(self, num_timesteps: int = 50, vocab_size: int = 10):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(self, x0, t):
        batch_size, height, width = x0.shape
        alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1)
        noise_mask = torch.rand(batch_size, height, width, device=x0.device) > alpha_bar
        random_tokens = torch.randint(0, self.vocab_size, (batch_size, height, width), device=x0.device)
        return torch.where(noise_mask, random_tokens, x0)


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

    def pad_tensor(tensor: torch.Tensor, fill_value: int):
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


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    input_grid = batch["input_grid"].to(device)
    output_grid = batch["output_grid"].to(device)
    output_mask = batch["output_mask"].to(device)
    task_ids = batch["task_idx"].to(device)
    heights = batch["height"].to(device)
    widths = batch["width"].to(device)
    d4_idx = batch["d4_idx"].to(device)
    color_shift = batch["color_shift"].to(device)

    return input_grid, output_grid, output_mask, task_ids, heights, widths, d4_idx, color_shift


def train_step(model, scheduler, batch, device):
    batch = pad_batch_to_model_size(batch, model.max_size)
    input_grid, output_grid, output_mask, task_ids, heights, widths, d4_idx, color_shift = batch_to_device(batch, device)

    batch_size = input_grid.shape[0]
    timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
    alpha_bar = scheduler.alpha_bars[timesteps].clamp(1e-5, 1 - 1e-5)
    logsnr = torch.log(alpha_bar) - torch.log(1 - alpha_bar)
    xt = scheduler.add_noise(output_grid, timesteps)

    with torch.no_grad():
        logits_prev = model(
            xt=xt,
            input_grid=input_grid,
            task_ids=task_ids,
            logsnr=logsnr,
            d4_idx=d4_idx,
            color_shift=color_shift,
            masks=output_mask,
            sc_p0=None,
        )
        sc = torch.log_softmax(logits_prev, dim=-1)
        sc = sc * output_mask.unsqueeze(-1).float()

    logits = model(
        xt=xt,
        input_grid=input_grid,
        task_ids=task_ids,
        logsnr=logsnr,
        d4_idx=d4_idx,
        color_shift=color_shift,
        masks=output_mask,
        sc_p0=sc,
    )

    per_cell_loss = F.cross_entropy(
        logits.view(-1, 10),
        output_grid.view(-1),
        reduction="none",
        ignore_index=10,
    )
    valid_mask = output_mask.view(-1).float()
    loss = (per_cell_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    predictions = logits.argmax(dim=-1)
    correct = ((predictions == output_grid) & output_mask).float().sum()
    acc = correct / output_mask.float().sum().clamp_min(1.0)

    return loss, acc


@torch.no_grad()
def evaluate(model, scheduler, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        if not batch["has_output"].any():
            continue

        batch = pad_batch_to_model_size(batch, model.max_size)
        input_grid, output_grid, output_mask, task_ids, heights, widths, d4_idx, color_shift = batch_to_device(batch, device)

        batch_size = input_grid.shape[0]
        timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        alpha_bar = scheduler.alpha_bars[timesteps].clamp(1e-5, 1 - 1e-5)
        logsnr = torch.log(alpha_bar) - torch.log(1 - alpha_bar)
        xt = scheduler.add_noise(output_grid, timesteps)

        logits_prev = model(
            xt=xt,
            input_grid=input_grid,
            task_ids=task_ids,
            logsnr=logsnr,
            d4_idx=d4_idx,
            color_shift=color_shift,
            masks=output_mask,
            sc_p0=None,
        )
        sc = torch.log_softmax(logits_prev, dim=-1)
        sc = sc * output_mask.unsqueeze(-1).float()

        logits = model(
            xt=xt,
            input_grid=input_grid,
            task_ids=task_ids,
            logsnr=logsnr,
            d4_idx=d4_idx,
            color_shift=color_shift,
            masks=output_mask,
            sc_p0=sc,
        )

        per_cell_loss = F.cross_entropy(
            logits.view(-1, 10),
            output_grid.view(-1),
            reduction="none",
            ignore_index=10,
        )
        valid_mask = output_mask.view(-1).float()
        total_loss += float((per_cell_loss * valid_mask).sum().item())
        total_correct += float((((logits.argmax(dim=-1) == output_grid) & output_mask).float().sum()).item())
        total_tokens += float(valid_mask.sum().item())

    mean_loss = total_loss / max(total_tokens, 1.0)
    mean_acc = total_correct / max(total_tokens, 1.0)
    return mean_loss, mean_acc


def build_parser():
    parser = argparse.ArgumentParser(description="Train ARC diffusion model using the ARC dataloader.")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max-size", type=int, default=30, help="Maximum grid size supported by the model")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--save-path", default="models/model.pt", help="Checkpoint path")
    parser.add_argument("--num-timesteps", type=int, default=50, help="Noise schedule steps")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional validation batch limit")
    parser.add_argument("--resume", default=None, help="Resume full training state from checkpoint")
    parser.add_argument(
        "--best-eval-acc-checkpoint",
        default=None,
        help="Initialize best eval accuracy from a checkpoint without resuming optimizer/epoch",
    )
    parser.add_argument("--log-dir", default="logs", help="Directory to save training logs and metrics")
    parser.add_argument("--log-interval", type=int, default=0, help="Log every N train batches; 0 disables")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory")
    return parser


def train():
    args = build_parser().parse_args()
    args = _apply_train_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    save_path = Path(args.save_path)
    if save_path.parent == Path("."):
        save_path = Path("models") / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_save_path = save_path
    model_run_dir = save_path.parent / f"run_{save_path.stem}_{run_tag}"
    model_run_dir.mkdir(parents=True, exist_ok=True)
    run_save_path = model_run_dir / save_path.name
    args.save_path = str(legacy_save_path)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{legacy_save_path.stem}_{run_tag}"
    log_path = log_dir / f"{run_name}.log"
    metrics_path = log_dir / f"{run_name}_metrics.jsonl"
    config_path = log_dir / f"{run_name}_config.json"
    logger = _setup_logger(log_path)
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    logger.info("Run config saved to %s", config_path)
    logger.info("Logging to %s", log_path)
    logger.info("Legacy checkpoints will be saved in %s", legacy_save_path.parent)
    logger.info("Model checkpoints for this run will be saved in %s", model_run_dir)

    tracker_path = Path("models") / "run_tracker.jsonl"
    tracker_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        train_batch_size=args.batch_size,
        test_batch_size=args.eval_batch_size,
        max_size=args.max_size,
        task_types=args.task_types,
        task_ids=args.task_ids,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    model = ARCDiffusionModel(max_size=args.max_size).to(device)
    scheduler = DiscreteNoiseScheduler(num_timesteps=args.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    param_stats = _model_param_stats(model)

    best_eval_acc = -1.0
    start_epoch = 0

    if args.best_eval_acc_checkpoint is not None:
        best_source = torch.load(args.best_eval_acc_checkpoint, map_location="cpu")
        if isinstance(best_source, dict):
            best_eval_acc = float(best_source.get("best_eval_acc", best_source.get("eval_acc", best_eval_acc)))
        logger.info("Initialized best_eval_acc=%.6f from %s", best_eval_acc, args.best_eval_acc_checkpoint)

    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=device)
        if isinstance(resume_ckpt, dict) and "model_state_dict" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model_state_dict"])
            if "optimizer_state_dict" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            start_epoch = int(resume_ckpt.get("epoch", 0))
            best_eval_acc = float(resume_ckpt.get("best_eval_acc", resume_ckpt.get("eval_acc", best_eval_acc)))
        else:
            model.load_state_dict(resume_ckpt)
            start_epoch = 0
        logger.info(
            "Resumed from %s: start_epoch=%d, best_eval_acc=%.6f",
            args.resume,
            start_epoch + 1,
            best_eval_acc,
        )

    if start_epoch >= args.epochs:
        logger.info(
            "Nothing to run: start_epoch=%d is already >= epochs=%d. Increase --epochs to continue training.",
            start_epoch,
            args.epochs,
        )
        return

    logger.info(
        "Starting training on %s from epoch %d to %d (batch_size=%d, eval_batch_size=%d)",
        device,
        start_epoch + 1,
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
    )

    _append_jsonl(
        tracker_path,
        {
            "mode": "train",
            "event": "run_started",
            "run_tag": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "save_path_legacy": str(legacy_save_path),
            "save_path_run": str(run_save_path),
            "resume": args.resume,
            "epochs_target": args.epochs,
            "start_epoch": start_epoch,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "data_dir": args.data_dir,
            **param_stats,
        },
    )

    run_status = "ok"
    error_message = None
    last_epoch = start_epoch
    try:
        try:
            total_train_batches = len(train_loader)
        except TypeError:
            total_train_batches = None

        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            train_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch_index, batch in enumerate(pbar):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break

                optimizer.zero_grad(set_to_none=True)
                loss, acc = train_step(model, scheduler, batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss_sum += float(loss.item())
                train_acc_sum += float(acc.item())
                train_batches += 1

                if args.log_interval > 0 and train_batches % args.log_interval == 0:
                    if total_train_batches is None:
                        logger.info(
                            "Epoch %d batch %d: loss=%.4f acc=%.4f",
                            epoch + 1,
                            train_batches,
                            loss.item(),
                            acc.item(),
                        )
                    else:
                        logger.info(
                            "Epoch %d batch %d/%d: loss=%.4f acc=%.4f",
                            epoch + 1,
                            train_batches,
                            total_train_batches,
                            loss.item(),
                            acc.item(),
                        )

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.3f}")

            mean_train_loss = train_loss_sum / max(train_batches, 1)
            mean_train_acc = train_acc_sum / max(train_batches, 1)

            eval_loss, eval_acc = evaluate(
                model,
                scheduler,
                test_loader,
                device,
                max_batches=args.max_eval_batches,
            )

            logger.info(
                "Epoch %d: train_loss=%.4f train_acc=%.3f eval_loss=%.4f eval_acc=%.3f",
                epoch + 1,
                mean_train_loss,
                mean_train_acc,
                eval_loss,
                eval_acc,
            )

            _append_jsonl(
                metrics_path,
                {
                    "epoch": epoch + 1,
                    "train_loss": mean_train_loss,
                    "train_acc": mean_train_acc,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "best_eval_acc_before_epoch": best_eval_acc,
                },
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "eval_acc": eval_acc,
                "best_eval_acc": max(best_eval_acc, eval_acc),
            }
            torch.save(checkpoint, legacy_save_path)
            torch.save(checkpoint, run_save_path)
            logger.info("Saved checkpoint (legacy): %s", legacy_save_path)
            logger.info("Saved checkpoint (run): %s", run_save_path)

            epoch_path = legacy_save_path.with_name(f"{legacy_save_path.stem}_epoch_{epoch + 1:03d}{legacy_save_path.suffix}")
            run_epoch_path = run_save_path.with_name(f"{run_save_path.stem}_epoch_{epoch + 1:03d}{run_save_path.suffix}")
            torch.save(checkpoint, epoch_path)
            torch.save(checkpoint, run_epoch_path)
            logger.info("Saved epoch checkpoint (legacy): %s", epoch_path)
            logger.info("Saved epoch checkpoint (run): %s", run_epoch_path)

            _append_jsonl(
                tracker_path,
                {
                    "mode": "train",
                    "event": "epoch_saved",
                    "run_tag": run_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "eval_acc": eval_acc,
                    "best_eval_acc": max(best_eval_acc, eval_acc),
                    "checkpoint_legacy": str(epoch_path),
                    "checkpoint_run": str(run_epoch_path),
                    "checkpoint_run_size_bytes": run_epoch_path.stat().st_size,
                    **param_stats,
                },
            )

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_path = legacy_save_path.with_name(f"best_{legacy_save_path.name}")
                run_best_path = run_save_path.with_name(f"best_{run_save_path.name}")
                torch.save(checkpoint, best_path)
                torch.save(checkpoint, run_best_path)
                logger.info("New best eval_acc=%.6f. Saved best checkpoint (legacy): %s", best_eval_acc, best_path)
                logger.info("New best eval_acc=%.6f. Saved best checkpoint (run): %s", best_eval_acc, run_best_path)

            last_epoch = epoch + 1
    except Exception as exc:
        run_status = "error"
        error_message = str(exc)
        logger.exception("Training failed with error: %s", exc)
        raise
    finally:
        _append_jsonl(
            tracker_path,
            {
                "mode": "train",
                "event": "run_finished",
                "run_tag": run_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epochs_target": args.epochs,
                "last_epoch": last_epoch,
                "final_best_eval_acc": best_eval_acc,
                "status": run_status,
                "error": error_message,
                **param_stats,
            },
        )


if __name__ == "__main__":
    train()