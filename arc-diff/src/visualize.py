"""ARC SVG visualizer with random sampling support."""

import argparse, html, json, random
from pathlib import Path

ARC_COLORS = {0: "#000000", 1: "#0074D9", 2: "#FF4136", 3: "#2ECC40", 4: "#FFDC00", 5: "#AAAAAA", 6: "#F012BE", 7: "#FF851B", 8: "#7FDBCA", 9: "#FFFFFF"}


class ARCVisualizer:
    def __init__(self, task_file):
        self.task_file = Path(task_file)
        self.data = json.loads(self.task_file.read_text(encoding="utf-8"))

    @staticmethod
    def _dims(grid):
        return len(grid[0]) if grid else 0, len(grid)

    def _grid_svg(self, grid, x, y, cell=24):
        return "\n".join(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{ARC_COLORS.get(int(v), "#000000")}" stroke="#333" stroke-width="1" />' for r, row in enumerate(grid) for c, v in enumerate(row))

    def _panel_svg(self, title, grid, x, y, cell=24):
        w, h = self._dims(grid)
        pw, ph = w * cell + 20, h * cell + 42
        parts = [f'<rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="8" ry="8" fill="#f7f7f7" stroke="#ccc" stroke-width="1.5" />', f'<text x="{x + 10}" y="{y + 18}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#111">{html.escape(title)}</text>']
        if grid: parts.append(self._grid_svg(grid, x + 10, y + 28, cell))
        return "\n".join(parts), pw, ph

    def _svg_markup(self):
        train, test = self.data.get("train", []), self.data.get("test", [])
        margin, gap_x, gap_y = 20, 20, 22
        rows = [f'<text x="{margin}" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#111">{html.escape(self.task_file.stem)}</text>']
        canvas_width = 640
        y = 60

        if train:
            rows.append(f'<text x="{margin}" y="{y - 14}" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#111">Training Examples</text>')
            y += 22
            for idx, ex in enumerate(train):
                inp_svg, inp_w, inp_h = self._panel_svg(f"Ex {idx + 1} Input", ex["input"], margin, y)
                out_x = margin + inp_w + gap_x
                out_svg, out_w, out_h = self._panel_svg(f"Ex {idx + 1} Output", ex["output"], out_x, y)
                rows.extend([inp_svg, out_svg])
                canvas_width = max(canvas_width, out_x + out_w + margin)
                y += max(inp_h, out_h) + gap_y

        if test:
            rows.append(f'<text x="{margin}" y="{y - 14}" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#111">Test Examples</text>')
            y += 22
            for idx, ex in enumerate(test):
                inp_svg, inp_w, inp_h = self._panel_svg(f"Test {idx + 1} Input", ex["input"], margin, y)
                if "output" in ex and ex["output"]:
                    out_x = margin + inp_w + gap_x
                    out_svg, out_w, out_h = self._panel_svg(f"Test {idx + 1} Output", ex["output"], out_x, y)
                    rows.extend([inp_svg, out_svg])
                    canvas_width = max(canvas_width, out_x + out_w + margin)
                    y += max(inp_h, out_h) + gap_y
                else:
                    rows.append(inp_svg)
                    canvas_width = max(canvas_width, margin + inp_w + margin)
                    y += inp_h + gap_y

        canvas_height = max(180, y + margin)
        return "\n".join(['<?xml version="1.0" encoding="UTF-8" standalone="no"?>', f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">', '<rect width="100%" height="100%" fill="#fff" />', *rows, "</svg>"])

    def save_svg(self, output_root="visuals", source_root=None):
        output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
        if source_root is None:
            rel_path = Path() if self.task_file.parent.name in {"", "."} else Path(self.task_file.parent.name)
        else:
            source_root = Path(source_root)
            try:
                rel_path = self.task_file.parent.relative_to(source_root) or Path(source_root.name)
            except ValueError:
                rel_path = Path(self.task_file.parent.name)
        target = output_root / rel_path / f"{self.task_file.stem}.svg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._svg_markup(), encoding="utf-8")
        return target

    def print_stats(self):
        train, test = self.data.get("train", []), self.data.get("test", [])
        print(f"\n{'=' * 50}"); print(f"File: {self.task_file.name}"); print(f"{'=' * 50}"); print(f"Training examples: {len(train)}"); print(f"Test examples: {len(test)}")
        if train:
            print("\nTraining input shapes:")
            for idx, ex in enumerate(train[:5]): inp, out = ex["input"], ex["output"]; print(f"  Example {idx + 1}: Input {len(inp)}x{len(inp[0]) if inp else 0} -> Output {len(out)}x{len(out[0]) if out else 0}")
            if len(train) > 5: print(f"  ... and {len(train) - 5} more")
        if test:
            print("\nTest input shapes:")
            for idx, ex in enumerate(test[:5]): inp = ex["input"]; print(f"  Example {idx + 1}: Input {len(inp)}x{len(inp[0]) if inp else 0}")
            if len(test) > 5: print(f"  ... and {len(test) - 5} more")


def iter_task_files(source_path):
    source_path = Path(source_path)
    if source_path.is_file():
        yield source_path
    else:
        yield from (p for p in sorted(source_path.rglob("*.json")) if p.is_file())


def select_task_files(source_path, random_count=None, seed=None):
    source_path = Path(source_path)
    if source_path.is_file(): return [source_path]
    files = list(iter_task_files(source_path))
    return files if random_count is None else random.Random(seed).sample(files, min(random_count, len(files)))


def visualize_arc(source, output_dir="visuals", random_count=None, seed=None, stats=False):
    source = Path(source)
    files = select_task_files(source, random_count=random_count, seed=seed)
    if not files: raise FileNotFoundError(f"No JSON task files found under: {source}")
    source_root = source if source.is_dir() else None
    for task_file in files:
        try: visualizer = ARCVisualizer(task_file)
        except json.JSONDecodeError as error:
            print(f"Skipping invalid JSON: {task_file} ({error})"); continue
        if stats: visualizer.print_stats()
        print(f"Saved SVG: {visualizer.save_svg(output_root=output_dir, source_root=source_root)}")


def build_parser():
    parser = argparse.ArgumentParser(description="Render ARC task files to SVG images.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""Examples:
  python src/visualize.py augmented_dataset/1d_mirror/1d_mirror_0.json
  python src/visualize.py augmented_dataset
  python src/visualize.py augmented_dataset --random 10 --stats""")
    parser.add_argument("source", help="Path to an ARC task JSON file or a folder of JSON files")
    parser.add_argument("--output-dir", default="visuals", help="Directory to write SVG files into")
    parser.add_argument("--stats", action="store_true", help="Print statistics before rendering")
    parser.add_argument("--random", type=int, dest="random_count", help="Render a random sample of files from a folder")
    parser.add_argument("--seed", type=int, help="Seed for random sampling")
    return parser


def main():
    args = build_parser().parse_args()
    visualize_arc(args.source, output_dir=args.output_dir, random_count=args.random_count, seed=args.seed, stats=args.stats)


if __name__ == "__main__": main()