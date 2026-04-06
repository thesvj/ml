import json
import matplotlib.pyplot as plt

def main():
    log_file = "/home/saij/ml/arc-diff/logs/model_20260404_020929_metrics.jsonl"
    
    epochs = []
    train_loss = []
    eval_loss = []
    
    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line)
            epochs.append(data["epoch"])
            train_loss.append(data["train_loss"])
            eval_loss.append(data.get("eval_loss", None))
            
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    
    # Only plot eval loss if it exists
    if any(loss is not None for loss in eval_loss):
        plt.plot(epochs, eval_loss, label='Eval Loss', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/saij/ml/arc-diff/logs/loss_curve.png')
    print("Saved loss curve to /home/saij/ml/arc-diff/logs/loss_curve.png")

if __name__ == "__main__":
    main()
