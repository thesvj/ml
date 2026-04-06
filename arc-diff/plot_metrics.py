import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    predictions_dir = "/home/saij/ml/arc-diff/predictions"
    pattern_wise_path = os.path.join(predictions_dir, "metrics_pattern_wise.json")
    summary_path = os.path.join(predictions_dir, "metrics_summary.json")

    # Load data
    with open(pattern_wise_path, "r") as f:
        pattern_data = json.load(f)
    
    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    # Get all patterns from dataset directory
    dataset_dir = "/home/saij/ml/arc-diff/dataset"
    if os.path.exists(dataset_dir):
        all_patterns = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    else:
        all_patterns = list(pattern_data.keys())

    # Extract pattern-wise metrics, defaulting to 0 for missing ones
    patterns = all_patterns
    task_accuracies = [pattern_data.get(p, {}).get("task_accuracy", 0) * 100 for p in patterns]
    cell_accuracies = [pattern_data.get(p, {}).get("cell_accuracy", 0) * 100 for p in patterns]

    # Overall Summary
    overall_task_acc = summary_data["task_accuracy"] * 100
    overall_cell_acc = summary_data["cell_accuracy"] * 100

    # Sort by task accuracy for better visualization
    sorted_indices = np.argsort(task_accuracies)
    patterns_sorted = [patterns[i] for i in sorted_indices]
    task_acc_sorted = [task_accuracies[i] for i in sorted_indices]
    cell_acc_sorted = [cell_accuracies[i] for i in sorted_indices]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 1]})

    # Plot 1: Pattern-wise history (Bar chart)
    y_pos = np.arange(len(patterns_sorted))
    height = 0.35

    rects1 = ax1.barh(y_pos - height/2, task_acc_sorted, height, label='Task Accuracy', color='skyblue')
    rects2 = ax1.barh(y_pos + height/2, cell_acc_sorted, height, label='Cell Accuracy', color='lightgreen')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(patterns_sorted)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Pattern-wise Accuracy Breakdown')
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Add text labels on bars
    for rect in rects1:
        width = rect.get_width()
        ax1.annotate(f'{width:.1f}%',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='left', va='center', size=8)
        
    for rect in rects2:
        width = rect.get_width()
        ax1.annotate(f'{width:.1f}%',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', size=8)

    # Plot 2: Summary visualization
    summary_labels = ['Total Task Accuracy', 'Total Cell Accuracy']
    summary_values = [overall_task_acc, overall_cell_acc]
    
    x_pos = np.arange(len(summary_labels))
    bars = ax2.bar(x_pos, summary_values, color=['dodgerblue', 'forestgreen'])
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(summary_labels)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Metrics Summary')
    ax2.set_ylim(0, 100)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    output_path = os.path.join(predictions_dir, "metrics_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()