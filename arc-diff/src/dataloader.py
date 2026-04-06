"""PyTorch DataLoaders for ARC train and test splits."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

class Dataset:
    """Lightweight protocol-style base class.

    This avoids hard dependency on torch at import time.
    """

    pass


def _get_torch_modules() -> Tuple[Any, Any]:
    try:
        torch_mod = importlib.import_module("torch")
        data_mod = importlib.import_module("torch.utils.data")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for DataLoader support. Install torch first.") from exc
    return torch_mod, data_mod.DataLoader


class ARCTrainTorchDataset(Dataset):
    """Torch-compatible dataset for ARC training examples."""

    def __init__(
        self,
        root_dir: str,
        max_size: int = 30,
        task_types: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.max_size = max_size
        self.task_types = set(task_types) if task_types else None
        self.task_ids = set(task_ids) if task_ids else None
        self.examples: List[Dict[str, object]] = []
        self.task_id_to_idx: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        for task_dir in sorted(self.root_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name
            if self.task_types is not None and task_name not in self.task_types:
                continue
            if task_name not in self.task_id_to_idx:
                self.task_id_to_idx[task_name] = len(self.task_id_to_idx)

            for file in sorted(task_dir.glob("*.json")):
                if self.task_ids is not None and file.stem not in self.task_ids:
                    continue
                with open(file, "r") as handle:
                    data = json.load(handle)

                for example in data.get("train", []):
                    input_grid = np.array(example["input"], dtype=np.int64)
                    output_grid = np.array(example["output"], dtype=np.int64)
                    if input_grid.shape[0] > self.max_size or input_grid.shape[1] > self.max_size:
                        continue
                    if output_grid.shape[0] > self.max_size or output_grid.shape[1] > self.max_size:
                        continue

                    self.examples.append({
                        "input": input_grid,
                        "output": output_grid,
                        "task_id": task_name,
                        "task_idx": self.task_id_to_idx[task_name],
                        "file_name": file.name,
                    })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        torch_mod, _ = _get_torch_modules()
        example = self.examples[idx]
        input_grid = example["input"]
        output_grid = example["output"]
        return {
            "input_grid": torch_mod.from_numpy(input_grid),
            "output_grid": torch_mod.from_numpy(output_grid),
            "task_idx": torch_mod.tensor(example["task_idx"], dtype=torch_mod.long),
            "height": torch_mod.tensor(output_grid.shape[0], dtype=torch_mod.long),
            "width": torch_mod.tensor(output_grid.shape[1], dtype=torch_mod.long),
            "d4_idx": torch_mod.tensor(0, dtype=torch_mod.long),
            "color_shift": torch_mod.tensor(0, dtype=torch_mod.long),
            "task_id": example["task_id"],
            "file_name": example["file_name"],
        }


class ARCTestTorchDataset(Dataset):
    """Torch-compatible dataset for ARC test examples."""

    def __init__(
        self,
        root_dir: str,
        max_size: int = 30,
        task_types: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.max_size = max_size
        self.task_types = set(task_types) if task_types else None
        self.task_ids = set(task_ids) if task_ids else None
        self.examples: List[Dict[str, object]] = []
        self.task_id_to_idx: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        for task_dir in sorted(self.root_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name
            if self.task_types is not None and task_name not in self.task_types:
                continue
            if task_name not in self.task_id_to_idx:
                self.task_id_to_idx[task_name] = len(self.task_id_to_idx)

            for file in sorted(task_dir.glob("*.json")):
                if self.task_ids is not None and file.stem not in self.task_ids:
                    continue
                with open(file, "r") as handle:
                    data = json.load(handle)

                for example in data.get("test", []):
                    output = example.get("output")
                    input_grid = np.array(example["input"], dtype=np.int64)
                    output_grid = None if output is None else np.array(output, dtype=np.int64)
                    if input_grid.shape[0] > self.max_size or input_grid.shape[1] > self.max_size:
                        continue
                    if output_grid is not None and (output_grid.shape[0] > self.max_size or output_grid.shape[1] > self.max_size):
                        continue

                    self.examples.append({
                        "input": input_grid,
                        "output": output_grid,
                        "task_id": task_name,
                        "task_idx": self.task_id_to_idx[task_name],
                        "file_name": file.name,
                    })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        torch_mod, _ = _get_torch_modules()
        example = self.examples[idx]
        input_grid = example["input"]
        output_grid = example["output"]
        return {
            "input_grid": torch_mod.from_numpy(input_grid),
            "output_grid": None if output_grid is None else torch_mod.from_numpy(output_grid),
            "task_idx": torch_mod.tensor(example["task_idx"], dtype=torch_mod.long),
            "height": torch_mod.tensor(-1 if output_grid is None else output_grid.shape[0], dtype=torch_mod.long),
            "width": torch_mod.tensor(-1 if output_grid is None else output_grid.shape[1], dtype=torch_mod.long),
            "d4_idx": torch_mod.tensor(0, dtype=torch_mod.long),
            "color_shift": torch_mod.tensor(0, dtype=torch_mod.long),
            "task_id": example["task_id"],
            "file_name": example["file_name"],
        }


def arc_collate_fn(batch: List[Dict], pad_value: int = 10) -> Dict[str, object]:
    """Pad variable-size ARC grids and stack into a batch."""

    torch_mod, _ = _get_torch_modules()

    max_h = 0
    max_w = 0
    for item in batch:
        in_h, in_w = item["input_grid"].shape
        max_h = max(max_h, in_h)
        max_w = max(max_w, in_w)
        if item["output_grid"] is not None:
            out_h, out_w = item["output_grid"].shape
            max_h = max(max_h, out_h)
            max_w = max(max_w, out_w)

    input_tensors: List[object] = []
    output_tensors: List[object] = []
    input_masks: List[object] = []
    output_masks: List[object] = []
    has_output: List[bool] = []
    task_ids: List[str] = []
    task_indices: List[object] = []
    heights: List[object] = []
    widths: List[object] = []
    d4_indices: List[object] = []
    color_shifts: List[object] = []
    file_names: List[str] = []

    for item in batch:
        inp = item["input_grid"].long()
        in_h, in_w = inp.shape

        padded_inp = torch_mod.full((max_h, max_w), pad_value, dtype=torch_mod.long)
        padded_inp[:in_h, :in_w] = inp

        input_mask = torch_mod.zeros((max_h, max_w), dtype=torch_mod.bool)
        input_mask[:in_h, :in_w] = True

        input_tensors.append(padded_inp)
        input_masks.append(input_mask)

        out = item["output_grid"]
        if out is None:
            padded_out = torch_mod.full((max_h, max_w), pad_value, dtype=torch_mod.long)
            output_mask = torch_mod.zeros((max_h, max_w), dtype=torch_mod.bool)
            has_output.append(False)
            heights.append(torch_mod.tensor(0, dtype=torch_mod.long))
            widths.append(torch_mod.tensor(0, dtype=torch_mod.long))
        else:
            out = out.long()
            out_h, out_w = out.shape
            padded_out = torch_mod.full((max_h, max_w), pad_value, dtype=torch_mod.long)
            padded_out[:out_h, :out_w] = out

            output_mask = torch_mod.zeros((max_h, max_w), dtype=torch_mod.bool)
            output_mask[:out_h, :out_w] = True
            has_output.append(True)
            heights.append(torch_mod.tensor(out_h, dtype=torch_mod.long))
            widths.append(torch_mod.tensor(out_w, dtype=torch_mod.long))

        output_tensors.append(padded_out)
        output_masks.append(output_mask)
        task_ids.append(item["task_id"])
        task_indices.append(item["task_idx"])
        d4_indices.append(item["d4_idx"])
        color_shifts.append(item["color_shift"])
        file_names.append(item["file_name"])

    return {
        "input_grid": torch_mod.stack(input_tensors, dim=0),
        "output_grid": torch_mod.stack(output_tensors, dim=0),
        "input_mask": torch_mod.stack(input_masks, dim=0),
        "output_mask": torch_mod.stack(output_masks, dim=0),
        "task_idx": torch_mod.stack(task_indices, dim=0),
        "height": torch_mod.stack(heights, dim=0),
        "width": torch_mod.stack(widths, dim=0),
        "d4_idx": torch_mod.stack(d4_indices, dim=0),
        "color_shift": torch_mod.stack(color_shifts, dim=0),
        "input_mask": torch_mod.stack(input_masks, dim=0),
        "has_output": torch_mod.tensor(has_output, dtype=torch_mod.bool),
        "task_ids": task_ids,
        "file_name": file_names,
    }


def get_train_dataloader(
    root_dir: str = "aug_data",
    batch_size: int = 32,
    max_size: int = 30,
    task_types: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    pad_value: int = 10,
) -> object:
    """Create DataLoader for ARC training split."""

    _, dataloader_cls = _get_torch_modules()
    dataset = ARCTrainTorchDataset(
        root_dir,
        max_size=max_size,
        task_types=task_types,
        task_ids=task_ids,
    )

    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: arc_collate_fn(b, pad_value=pad_value),
    )


def get_test_dataloader(
    root_dir: str = "aug_data",
    batch_size: int = 32,
    max_size: int = 30,
    task_types: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    pad_value: int = 10,
) -> object:
    """Create DataLoader for ARC test split."""

    _, dataloader_cls = _get_torch_modules()
    dataset = ARCTestTorchDataset(
        root_dir,
        max_size=max_size,
        task_types=task_types,
        task_ids=task_ids,
    )

    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: arc_collate_fn(b, pad_value=pad_value),
    )


def get_dataloaders(
    root_dir: str = "aug_data",
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    max_size: int = 30,
    task_types: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    pad_value: int = 10,
) -> Tuple[object, object]:
    """Create both ARC train and test DataLoaders."""

    train_loader = get_train_dataloader(
        root_dir=root_dir,
        batch_size=train_batch_size,
        max_size=max_size,
        task_types=task_types,
        task_ids=task_ids,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pad_value=pad_value,
    )

    test_loader = get_test_dataloader(
        root_dir=root_dir,
        batch_size=test_batch_size,
        max_size=max_size,
        task_types=task_types,
        task_ids=task_ids,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pad_value=pad_value,
    )

    return train_loader, test_loader


def _summarize_batch(name: str, batch: Dict[str, object]) -> None:
    print(f"{name} batch keys: {list(batch.keys())}")
    print(f"{name} input shape: {tuple(batch['input'].shape)}")
    print(f"{name} output shape: {tuple(batch['output'].shape)}")
    print(f"{name} input_mask shape: {tuple(batch['input_mask'].shape)}")
    print(f"{name} output_mask shape: {tuple(batch['output_mask'].shape)}")
    print(f"{name} has_output shape: {tuple(batch['has_output'].shape)}")
    print(f"{name} sample task_type: {batch['task_type'][0] if batch['task_type'] else ''}")
    print(f"{name} sample file_name: {batch['file_name'][0] if batch['file_name'] else ''}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and smoke-test ARC train/test DataLoaders.")
    parser.add_argument("--root-dir", default="aug_data", help="Directory with ARC JSON task folders")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Test batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pad-value", type=int, default=10, help="Grid pad token")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    train_loader, test_loader = get_dataloaders(
        root_dir=args.root_dir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        pad_value=args.pad_value,
    )

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    _summarize_batch("train", train_batch)
    _summarize_batch("test", test_batch)


if __name__ == "__main__":
    main()
