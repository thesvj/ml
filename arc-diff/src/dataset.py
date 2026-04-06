import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================
# 1. AUGMENTATION UTILITIES
# ============================================================

class TaskAugmentation:
    """Applies D4 symmetry + color transformations."""

    @staticmethod
    def apply_d4(grid: np.ndarray, d4_idx: int) -> np.ndarray:
        if d4_idx == 0:
            return grid
        elif d4_idx == 1:
            return np.rot90(grid, 1)
        elif d4_idx == 2:
            return np.rot90(grid, 2)
        elif d4_idx == 3:
            return np.rot90(grid, 3)
        elif d4_idx == 4:
            return np.fliplr(grid)
        elif d4_idx == 5:
            return np.flipud(grid)
        elif d4_idx == 6:
            return np.transpose(grid)
        elif d4_idx == 7:
            return np.fliplr(np.transpose(grid))
        else:
            raise ValueError(f"Invalid d4_idx: {d4_idx}")

    @staticmethod
    def apply_color_shift(grid: np.ndarray, shift: int) -> np.ndarray:
        return (grid + shift) % 10  # ARC colors: 0–9


# ============================================================
# 2. AUGMENTATION GENERATOR
# ============================================================

def generate_augmentations(remove_identity: bool = True) -> List[Tuple[int, int]]:
    d4 = list(range(8))
    colors = list(range(10))

    augmentations = [(d, c) for d in d4 for c in colors]

    if remove_identity:
        augmentations = [(d, c) for d, c in augmentations if not (d == 0 and c == 0)]

    return augmentations


# ============================================================
# 3. DATASET BUILDER
# ============================================================

class ARCDatasetBuilder:
    """
    Converts folder-based ARC dataset into augmented dataset.

    Input format:
        dataset/
            task_type/
                task.json

    Output format:
        output/
            task_type/
                task.json  (augmented)
    """

    def __init__(
        self,
        input_root: str,
        output_root: str,
        augment: bool = True,
        max_per_task: int = None,
        augment_test: bool = False
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.augment = augment
        self.max_per_task = max_per_task
        self.augment_test = augment_test

        self.augmentations = generate_augmentations()

    # --------------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------------

    def build(self):
        print(f"Processing dataset from: {self.input_root}")
        print(f"Saving to: {self.output_root}")

        for task_dir in self.input_root.iterdir():
            if not task_dir.is_dir():
                continue

            out_dir = self.output_root / task_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for file in task_dir.glob("*.json"):
                self._process_file(file, out_dir)

        print("Dataset build complete.")

    # --------------------------------------------------------
    # PROCESS SINGLE FILE
    # --------------------------------------------------------

    def _process_file(self, file_path: Path, out_dir: Path):
        with open(file_path, "r") as f:
            data = json.load(f)

        train = data.get("train", [])
        test = data.get("test", [])

        train_np = [(np.array(x["input"]), np.array(x["output"])) for x in train]
        
        # Separate test examples into those with output and those without
        test_with_output = [(np.array(x["input"]), np.array(x["output"])) for x in test if "output" in x and x["output"]]
        test_without_output = [np.array(x["input"]) for x in test if "output" not in x or not x["output"]]

        new_train = []
        new_test = []

        # ---------------------------
        # Add original data
        # ---------------------------
        for inp, out in train_np:
            new_train.append(self._to_dict(inp, out))

        for inp, out in test_with_output:
            new_test.append(self._to_dict(inp, out))
        
        for inp in test_without_output:
            new_test.append({"input": inp.tolist()})

        # ---------------------------
        # Apply augmentation
        # ---------------------------
        if self.augment:
            for d4_idx, color_shift in self.augmentations:

                # TRAIN AUGMENTATION
                for inp, out in train_np:
                    aug_inp = TaskAugmentation.apply_d4(inp, d4_idx)
                    aug_inp = TaskAugmentation.apply_color_shift(aug_inp, color_shift)

                    aug_out = TaskAugmentation.apply_d4(out, d4_idx)
                    aug_out = TaskAugmentation.apply_color_shift(aug_out, color_shift)

                    new_train.append(self._to_dict(aug_inp, aug_out))

                # TEST AUGMENTATION (optional) - augment both input and output when output exists
                if self.augment_test:
                    for inp, out in test_with_output:
                        aug_inp = TaskAugmentation.apply_d4(inp, d4_idx)
                        aug_inp = TaskAugmentation.apply_color_shift(aug_inp, color_shift)

                        aug_out = TaskAugmentation.apply_d4(out, d4_idx)
                        aug_out = TaskAugmentation.apply_color_shift(aug_out, color_shift)

                        new_test.append(self._to_dict(aug_inp, aug_out))
                    
                    for inp in test_without_output:
                        aug_inp = TaskAugmentation.apply_d4(inp, d4_idx)
                        aug_inp = TaskAugmentation.apply_color_shift(aug_inp, color_shift)
                        new_test.append({"input": aug_inp.tolist()})

        # ---------------------------
        # Limit dataset size
        # ---------------------------
        if self.max_per_task is not None:
            new_train = new_train[:self.max_per_task]

        # ---------------------------
        # Save output
        # ---------------------------
        output_data = {
            "train": new_train,
            "test": new_test
        }

        out_path = out_dir / file_path.name
        with open(out_path, "w") as f:
            json.dump(output_data, f)

        print(f"Processed: {file_path.name} | Train: {len(new_train)}")

    # --------------------------------------------------------
    # UTILITY
    # --------------------------------------------------------

    def _to_dict(self, inp: np.ndarray, out: np.ndarray):
        return {
            "input": inp.tolist(),
            "output": out.tolist()
        }


# ============================================================
# 4. SIMPLE DATASET (OPTIONAL FOR TRAINING)
# ============================================================

class SimpleARCDataset:
    """Loads augmented dataset for training."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.examples = []

        self._load()

    def _load(self):
        for task_dir in self.root_dir.iterdir():
            if not task_dir.is_dir():
                continue

            for file in task_dir.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)

                for ex in data["train"]:
                    self.examples.append((
                        np.array(ex["input"]),
                        np.array(ex["output"])
                    ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SimpleARCEvalDataset:
    """Loads test split for evaluation."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.examples = []

        self._load()

    def _load(self):
        for task_dir in self.root_dir.iterdir():
            if not task_dir.is_dir():
                continue

            for file in task_dir.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)

                for ex in data.get("test", []):
                    output = ex.get("output")
                    self.examples.append((
                        np.array(ex["input"]),
                        np.array(output) if output else None,
                        task_dir.name,
                        file.name,
                    ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], str, str]:
        return self.examples[idx]


# ============================================================
# 5. USAGE
# ============================================================

if __name__ == "__main__":

    builder = ARCDatasetBuilder(
        input_root="dataset",
        output_root="aug_data",
        augment=True,
        max_per_task=500,       # prevent explosion
        augment_test=True       # augment test examples too
    )

    builder.build()

    # Optional: load dataset
    dataset = SimpleARCDataset("aug_data")
    print(f"Total training examples: {len(dataset)}")

    eval_dataset = SimpleARCEvalDataset("aug_data")
    print(f"Total evaluation examples: {len(eval_dataset)}")