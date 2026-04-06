"""Example usage of the ARCVisualizer class.

The visualizer now writes SVG files instead of opening a GUI.
"""

from pathlib import Path

from src.visualize import ARCVisualizer

# ============================================================
# Example 1: Quick statistics of a single file
# ============================================================

def example_stats():
    """Print statistics for a task file."""
    viz = ARCVisualizer('augmented_dataset/1d_mirror/1d_mirror_0.json')
    viz.print_stats()


# ============================================================
# Example 2: Render a single task file
# ============================================================

def example_render_single():
    """Render a single task file to visuals/."""
    viz = ARCVisualizer('augmented_dataset/1d_fill/1d_fill_5.json')
    output_path = viz.save_svg(output_root='visuals')
    print(f'Rendered: {output_path}')


# ============================================================
# Example 3: Batch render a folder
# ============================================================

def example_batch_inspect():
    """Render an entire task folder and print statistics."""
    task_dir = Path('augmented_dataset/1d_mirror')
    for filepath in sorted(task_dir.glob('*.json'))[:5]:
        print(f"\n{'=' * 60}")
        viz = ARCVisualizer(filepath)
        viz.print_stats()
        print(f"Rendered: {viz.save_svg(output_root='visuals', source_root=task_dir)}")


# ============================================================
# Example 6: Custom analysis - find interesting patterns
# ============================================================

def example_analysis():
    """Analyze grid properties of multiple files."""
    import json
    import os
    from pathlib import Path
    
    task_dir = 'augmented_dataset'
    stats = {
        'total_files': 0,
        'total_train': 0,
        'total_test': 0,
        'avg_input_size': [],
        'tasks': {}
    }
    
    for root, dirs, files in os.walk(task_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                train_data = data.get('train', [])
                test_data = data.get('test', [])
                
                task_name = Path(root).name
                
                stats['total_files'] += 1
                stats['total_train'] += len(train_data)
                stats['total_test'] += len(test_data)
                
                if task_name not in stats['tasks']:
                    stats['tasks'][task_name] = {'train': 0, 'test': 0, 'files': 0}
                
                stats['tasks'][task_name]['train'] += len(train_data)
                stats['tasks'][task_name]['test'] += len(test_data)
                stats['tasks'][task_name]['files'] += 1
                
                # Analyze input sizes
                if train_data:
                    inp = train_data[0]['input']
                    size = len(inp) * len(inp[0])
                    stats['avg_input_size'].append(size)
    
    # Print report
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total training examples: {stats['total_train']}")
    print(f"Total test examples: {stats['total_test']}")
    print(f"Average input grid size: {sum(stats['avg_input_size'])/len(stats['avg_input_size']):.0f} cells")
    
    print(f"\nPer-task statistics:")
    for task_name, task_stats in sorted(stats['tasks'].items()):
        print(f"  {task_name:20s}: {task_stats['files']:2d} files, "
              f"{task_stats['train']:6d} train, {task_stats['test']:3d} test")


def example_compare():
    """Compare statistics between two task files."""
    file1 = 'augmented_dataset/1d_mirror/1d_mirror_0.json'
    file2 = 'augmented_dataset/1d_fill/1d_fill_0.json'
    
    print(f"Comparing {file1} vs {file2}\n")
    
    for filepath in [file1, file2]:
        viz = ARCVisualizer(filepath)
        viz.print_stats()


# ============================================================
# MAIN - Uncomment the example you want to run
# ============================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        'stats': example_stats,
        'render': example_render_single,
        'batch': example_batch_inspect,
        'analysis': example_analysis,
        'compare': example_compare,
    }
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name in examples:
            print(f"Running: {example_name}")
            examples[example_name]()
        else:
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        # Run analysis by default
        print("Running dataset analysis (pass example name as argument to run other examples)")
        examples['analysis']()
