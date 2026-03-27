# Numin - Meta-Learning for OHLCV-to-Returns Prediction

A meta-learning framework for predicting daily returns from OHLCV (Open, High, Low, Close, Volume) data. This project adapts meta-learning techniques from computer vision (similar to ARC-Meta) to financial time-series forecasting.

## Task

**Goal:** Given n-1 days of OHLCV data for multiple assets, predict the returns for day n.

This is a time-series regression task where the model learns patterns in price movements and trading volume to forecast future returns using a meta-learning approach.

## Project Structure

```
numin/
├── src/                       # Core modules
│   ├── __init__.py
│   ├── dataset.py             # Dataset classes (standard + meta-learning)
│   ├── dataloader.py          # DataLoader utilities
│   ├── model.py               # Meta-learning model architecture
│   └── eval.py                # Evaluation script
├── data/                      # Data files
│   ├── consolidated_daily_ohlcv.parquet
│   ├── consolidated_daily_returns.parquet
│   ├── consolidated_daily_ohlcv_last_10.csv
│   └── consolidated_daily_returns_last_10.csv
├── .venv/                     # uv virtual environment
├── train.py                   # Training script
├── test_demo.py               # Validation/testing script
├── setup.sh                   # Quick setup helper
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

Generated directories (created during training):
- `models/` - Trained model checkpoints
- `logs/` - Training logs and metrics


## Meta-Learning Architecture

The model implements a meta-learning approach inspired by ARC-Meta but adapted for time-series data:

### Components

1. **SequenceEncoder** - Encodes OHLCV sequences
   - Bidirectional LSTM for temporal feature extraction
   - Self-attention mechanism for sequence aggregation
   - Produces rich feature representations

2. **TaskEncoder** - Learns task-specific representations
   - Combines support input sequences with their target returns
   - Cross-attention across support samples
   - Extracts patterns relevant for return prediction

3. **PredictorHead** - Predicts returns using task context
   - Cross-attention between query sequences and task context
   - Multi-layer feed-forward for final prediction
   - Regression output (continuoususally 1 value per asset/timestamp)

### Forward Pass

```
Support + Query Sequences
        ↓
   SequenceEncoder (shared)
        ↓
Support Features → TaskEncoder → Task Tokens
        ↓
Query Features + Task Tokens → PredictorHead → Predictions
```

## Installation

### Prerequisites
- Python 3.12+
- CUDA 11.8+ (optional, for GPU acceleration)
- `uv` package manager

### Quick Setup (Recommended)

```bash
cd numin
bash setup.sh
```

### Manual Setup

```bash
cd numin

# Create virtual environment with uv
uv venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```


## Usage

### 1. Standard Supervised Learning (No Meta-Learning)

```bash
python train.py \
    --lookback 10 \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 100 \
    --hidden_size 128
```

### 2. Meta-Learning Mode

```bash
python train.py \
    --meta \
    --lookback 10 \
    --batch_size 16 \
    --k_shot 5 \
    --q_query 1 \
    --lr 0.001 \
    --epochs 100 \
    --hidden_size 128
```

### Training Arguments

- `--meta`: Enable meta-learning mode
- `--lookback`: Number of days to look back (default: 10)
- `--batch_size`: Batch size (tasks for meta-learning, samples for standard)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Training epochs (default: 100)
- `--hidden_size`: Model dimension (default: 128)
- `--num_heads`: Attention heads (default: 4)
- `--k_shot`: Support samples per meta-task (default: 5)
- `--q_query`: Query samples per meta-task (default: 1)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--resume`: Checkpoint path to resume training
- `--seed`: Random seed (default: 42)

### 3. Evaluation

```bash
python src/eval.py \
    --checkpoint models/meta_ohlcv_meta_best.pt \
    --meta \
    --lookback 10 \
    --k_shot 5 \
    --q_query 1
```

## Meta-Learning vs Standard Training

### Standard Mode
- Treats each sequence independently
- Direct supervised learning: sequence → returns
- Simpler, faster training
- Best for: Single-asset prediction, stable patterns

### Meta-Learning Mode
- Groups sequences into tasks (support + query)
- Learns task-specific representations
- Learns how to adapt to different market conditions
- Better generalization to unseen assets
- Best for: Multi-asset prediction, market regime changes

## Data Format

The system expects two parquet files:
- `consolidated_daily_ohlcv.parquet`: Shape (n_days, n_features)
- `consolidated_daily_returns.parquet`: Shape (n_days, n_returns)

Data splits:
- **Train:** 70%
- **Val:** 20% of remaining
- **Test:** 10%

## Model Details

### Input Sizes
- Sequences: (batch, time_steps, 5) - for OHLCV data
- Returns: (batch, num_assets)

### Output
- Predictions: (batch, num_assets) - predicted returns

### Computational Requirements
- GPU Memory: ~2-4 GB (depending on batch size)
- Training Time: ~2-5 minutes per epoch on RTX 3090
- Parameters: ~500K-1.5M depending on hidden_size

## Quick Start

```bash
# 1. Navigate to numin folder
cd numin

# 2. Run setup
bash setup.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Validate setup (optional)
python test_demo.py

# 5. Train meta-learning model
python train.py --meta --epochs 50

# 6. Evaluate
python src/eval.py --checkpoint models/meta_ohlcv_meta_best.pt --meta

# 7. Check results
cat logs/eval_*.json
```

## Hyperparameter Tuning

### For Better Performance

**If training is unstable (high variance in loss):**
- Reduce `--lr` (try 1e-4, 5e-4)
- Increase `--weight_decay` (try 1e-4, 1e-3)
- Try `--k_shot 10` for more support samples

**If model is underfitting (high train + val loss):**
- Increase `--hidden_size` (try 256, 512)
- Increase `--epochs`
- Increase `--k_shot` for more diverse support sets

**If model is overfitting (low train, high val loss):**
- Increase `--weight_decay`
- Reduce `--batch_size`
- Reduce `--hidden_size`

**For faster meta-learning:**
- Reduce `--k_shot` (trade quality for speed)
- Reduce `--lookback` (shorter sequences)

## Expected Performance

With default hyperparameters on typical financial data:
- **MSE:** 0.01-0.05
- **RMSE:** 0.1-0.22
- **MAE:** 0.08-0.18
- **R²:** 0.1-0.4

Note: Financial data is inherently noisy. Performance depends heavily on data quality and market conditions.

## Advanced: Custom Meta-Learning Tasks

```python
from src.dataset import OHLCVReturnsDataset
from src.dataloader import get_dataloaders_meta

# Create meta-learning dataset
dataset = OHLCVReturnsDataset(
    data_dir="data",
    lookback=10,
    split='train',
    meta_learning=True,
    k_shot=5,      # 5 support samples
    q_query=2      # 2 query samples
)

# Use with dataloader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Get a task
task = next(iter(loader))
support_x = task['support_x']  # (B, K, T, features)
support_y = task['support_y']  # (B, K, returns)
query_x = task['query_x']      # (B, Q, T, features)
query_y = task['query_y']      # (B, Q, returns)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch_size` or `--hidden_size` |
| NaN in loss | Reduce `--lr`, check data for NaNs |
| High val loss | Increase regularization, reduce model complexity |
| Slow training | Use GPU, reduce `--lookback`, reduce `--batch_size` |

## Validation

Run `test_demo.py` to verify the installation and that both standard and meta-learning modes work correctly:

```bash
python test_demo.py
```

This will:
- Test data loading for both standard and meta-learning modes
- Create and validate the model architecture
- Run a forward pass to ensure everything works
- Report any issues with setup

## References

- Meta-Learning: Finn et al., "Model-Agnostic Meta-Learning" (MAML)
- ARC-Meta: Meta-Learning for ARC tasks (similar architecture)
- Attention: Vaswani et al., "Attention is All You Need"
- LSTM: Hochreiter & Schmidhuber (1997)

## Future Improvements

1. **Multi-horizon prediction**: Predict 1-day, 5-day, 10-day returns
2. **Task-aware adaptation**: Fine-tune on new assets quickly
3. **Attention visualization**: Understand which features matter
4. **Ensemble methods**: Combine multiple models
5. **Technical indicators**: Add RSI, MACD, Bollinger Bands

## License

MIT
