import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os

# =============================================================================
# 1. MODEL ARCHITECTURE (Redefined from training script)
# =============================================================================
class MultiWindowFeatureExtractor(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.short_conv = nn.Conv1d(input_dims, output_dims, kernel_size=3, padding=1)
        self.mid_conv = nn.Conv1d(input_dims, output_dims, kernel_size=5, padding=2)
        self.long_conv = nn.Conv1d(input_dims, output_dims, kernel_size=7, padding=3)
        self.batch_norm = nn.BatchNorm1d(output_dims * 3)
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, x):
        out_short = self.short_conv(x)
        out_mid   = self.mid_conv(x)
        out_long  = self.long_conv(x)
        combined = torch.cat([out_short, out_mid, out_long], dim=1)
        return self.dropout_layer(F.gelu(self.batch_norm(combined)))

class StockPricePredictor(nn.Module):
    def __init__(self, num_features, hidden_size=64):
        super().__init__()
        self.initial_norm = nn.BatchNorm1d(num_features)
        self.feature_block = MultiWindowFeatureExtractor(num_features, hidden_size)
        self.compression = nn.Conv1d(hidden_size * 3, hidden_size, kernel_size=1)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.initial_norm(x)
        x = self.feature_block(x)
        x = F.gelu(self.compression(x))
        avg_pool = torch.mean(x, dim=-1)
        max_pool, _ = torch.max(x, dim=-1)
        pooled_output = (avg_pool + max_pool) / 2
        return self.output_head(pooled_output).squeeze(-1)


# =============================================================================
# 2. INFERENCE PREPARATION
# =============================================================================
def prepare_inference_data(data_path, ticker='RELIANCE', seq_len=30):
    """
    Loads historical data, engineers the required 13 features, and extracts exactly 
    the last `seq_len` days to form a single tensor matching what the model expects.
    """
    df = pd.read_csv(data_path)
    df = df[df['ticker'] == ticker].copy()
    df.sort_values('timestamp', inplace=True)
    
    # Feature Engineering (Same logic as in train.ipynb)
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['ma_5'] = df['close'].rolling(5).mean() / df['close'] - 1
    df['ma_10'] = df['close'].rolling(10).mean() / df['close'] - 1
    df['ma_20'] = df['close'].rolling(20).mean() / df['close'] - 1
    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    df['vol_change'] = df['volume'].pct_change(1)
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    
    df.dropna(inplace=True)
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'return_1d', 'return_5d', 'ma_5', 'ma_10', 'ma_20', 
                    'volatility_5d', 'vol_change', 'hl_spread']
    
    # Normalization
    features = df[feature_cols].values
    
    # DANGER WARNING: In a purely strict ML pipeline, you would load the exact `mean` 
    # and `std` from when you trained your dataset. 
    # For quick inference demo, we normalize using the latest batch window.
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / (std + 1e-8)
    
    if len(features) < seq_len:
        raise ValueError(f"Not enough data for ticker {ticker}. Need at least {seq_len} days, got {len(features)}.")
    
    # Grab the most recent window of length `seq_len` to predict tomorrow
    latest_window = features[-seq_len:]
    
    # Add a batch dimension to match what the model needs: shape (1, seq_len, num_features)
    input_tensor = torch.tensor(latest_window, dtype=torch.float32).unsqueeze(0)
    
    last_close = df['close'].iloc[-1]
    last_timestamp = df['timestamp'].iloc[-1]
    
    return input_tensor, last_close, last_timestamp


# =============================================================================
# 3. PREDICT FUNCTION
# =============================================================================
def predict_next_day(model_path, data_path, ticker='RELIANCE', seq_len=30, features_count=13):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Preparing latest data for {ticker}...")
    try:
        input_tensor, last_close, last_date = prepare_inference_data(data_path, ticker, seq_len)
    except Exception as e:
        print(f"Error prepping data: {e}")
        return
        
    input_tensor = input_tensor.to(device)
    
    print(f"Loading Model architecture on [{device}]...")
    model = StockPricePredictor(num_features=features_count)
    model.to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained weights from {model_path}")
    else:
        print(f"⚠️ Weights file '{model_path}' not found! The model will make a RANDOM prediction.")
        
    model.eval()
    with torch.no_grad():
        predicted_return = model(input_tensor).item()
        
    predicted_price = last_close * (1 + predicted_return)
    
    print("\n" + "="*50)
    print(f" 🔮 INFERENCE RESULTS FOR {ticker} 🔮 ")
    print("="*50)
    print(f"Last Data Date         : {last_date}")
    print(f"Last Close Price       : {last_close:.2f} INR")
    print(f"Predicted Return       : {predicted_return * 100:.3f}%")
    print(f"Predicted Next Price   : {predicted_price:.2f} INR")
    print("="*50)


if __name__ == "__main__":
    # Ensure this paths match your current file structure relative to execution domain
    DATA_PATH = "../data/dataset_clean.csv"
    MODEL_PATH = "../reliance_model.pth"
    
    predict_next_day(
        model_path=MODEL_PATH, 
        data_path=DATA_PATH, 
        ticker='RELIANCE', 
        seq_len=30, 
        features_count=13
    )