import numpy as np

def rolling_mean(x, window):
    """Calculates rolling mean along axis 0 using cumsum for speed."""
    T, N = x.shape
    # Pad with NaNs to maintain shape (T, N) matching standard rolling behavior
    result = np.full((T, N), np.nan)
    
    # Cumsum trick for fast rolling window
    # We pad x with a row of zeros at the top to make vectorization easy
    pad = np.zeros((1, N))
    x_padded = np.vstack([pad, x])
    cumsum = np.cumsum(x_padded, axis=0)
    
    # The rolling sum is cumsum[i] - cumsum[i-window]
    rolling_sum = cumsum[window:] - cumsum[:-window]
    
    # Fill valid range
    result[window-1:] = rolling_sum / window
    return result

def rolling_std(x, window):
    """Calculates rolling std dev using E[X^2] - E[X]^2 identity."""
    # Note: This method is fast but can have small numerical precision errors.
    # For production finance, consider a 2-pass algorithm or pandas.
    mean = rolling_mean(x, window)
    mean_sq = rolling_mean(x**2, window)
    
    # Clip negative values due to floating point errors
    var = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(var)

def rolling_max(x, window):
    """Calculates rolling max using a loop (NumPy doesn't have a fast native rolling_max)."""
    T, N = x.shape
    result = np.full((T, N), np.nan)
    
    # Basic sliding window loop
    for i in range(window - 1, T):
        result[i] = np.max(x[i-window+1 : i+1], axis=0)
        
    return result

def rolling_corr(x, y, window):
    """Calculates rolling correlation between matrix x (T, N) and vector y (T, 1)."""
    # Ensure y is broadcastable (T, 1)
    if y.ndim == 1:
        y = y[:, np.newaxis]
        
    mean_x = rolling_mean(x, window)
    mean_y = rolling_mean(y, window) 
    mean_xy = rolling_mean(x * y, window)
    
    cov = mean_xy - mean_x * mean_y
    std_x = rolling_std(x, window)
    std_y = rolling_std(y, window)
    
    # Add tiny epsilon to avoid division by zero
    return cov / (std_x * std_y + 1e-8)

def compute_features(open_, high, low, close, volume):
    """
    Computes financial features (Momentum, Volatility, Relative Strength).
    
    Args:
        open_, high, low, close, volume: np.arrays of shape (T, N)
        
    Returns:
        features: np.array of shape (T-30, N, 13)
                  (First 30 rows dropped to remove NaN artifacts)
    """
    eps = 1e-6
    T, N = close.shape
    
    # =========================
    # 1. RETURNS
    # =========================
    # Log returns for additivity
    ret_1 = np.log(close[1:] / (close[:-1] + eps))
    ret_1 = np.vstack([np.zeros((1, N)), ret_1])

    ret_5 = np.log(close[5:] / (close[:-5] + eps))
    ret_5 = np.vstack([np.zeros((5, N)), ret_5])

    ret_30 = np.log(close[30:] / (close[:-30] + eps))
    ret_30 = np.vstack([np.zeros((30, N)), ret_30])

    # =========================
    # 2. MARKET RETURN
    # =========================
    market_ret_5 = np.mean(ret_5, axis=1, keepdims=True)
    market_ret_1 = np.mean(ret_1, axis=1, keepdims=True)

    # =========================
    # 3. RELATIVE + RANK
    # =========================
    rel_ret_5 = ret_5 - market_ret_5

    # Normalize rank [0, 1]
    # Use argsort twice to get ranks
    rank_ret_5 = np.argsort(np.argsort(ret_5, axis=1), axis=1) / (N - 1 + eps)

    # =========================
    # 4. VOLATILITY
    # =========================
    vol_30 = rolling_std(ret_1, 30)
    risk_adj_ret = ret_5 / (vol_30 + eps)

    # =========================
    # 5. VOLUME FEATURES
    # =========================
    vol_mean_20 = rolling_mean(volume, 20)
    vol_std_20  = rolling_std(volume, 20)

    vol_z = (volume - vol_mean_20) / (vol_std_20 + eps)
    pv_signal = ret_1 * vol_z

    # =========================
    # 6. STRUCTURE
    # =========================
    rolling_high_20 = rolling_max(close, 20)
    dist_high = (close - rolling_high_20) / (close + eps)

    mean_20 = rolling_mean(close, 20)
    std_20  = rolling_std(close, 20)
    z_price = (close - mean_20) / (std_20 + eps)

    # =========================
    # 7. RESIDUAL (BETA)
    # =========================
    cov = rolling_mean(ret_1 * market_ret_1, 30) - \
          rolling_mean(ret_1, 30) * rolling_mean(market_ret_1, 30)

    var_market = rolling_std(market_ret_1, 30) ** 2

    beta = cov / (var_market + eps)
    residual = ret_5 - beta * market_ret_5

    # =========================
    # 8. CORRELATION
    # =========================
    corr_market_30 = rolling_corr(ret_1, market_ret_1, 30)

    # =========================
    # 9. STACK FEATURES
    # =========================
    # Repeat market return to match shape (T, N)
    market_ret_5_expanded = np.tile(market_ret_5, (1, N))

    features = np.stack([
        ret_5,
        ret_30,
        rel_ret_5,
        rank_ret_5,
        vol_30,
        risk_adj_ret,
        vol_z,
        pv_signal,
        dist_high,
        z_price,
        residual,
        market_ret_5_expanded,
        corr_market_30
    ], axis=-1)

    # Drop the first 30 rows (warmup period) to avoid NaNs/Zeros
    return features[30:]

if __name__ == "__main__":
    # --- Quick Test ---
    print("Generating dummy data...")
    T, N = 100, 50  # 100 timesteps, 50 assets
    
    # Random walk prices
    close = np.cumsum(np.random.randn(T, N), axis=0) + 100
    # Dummy volume
    volume = np.abs(np.random.randn(T, N) * 1000) + 100
    
    # Open, high, low just duplicates for this test
    feats = compute_features(close, close, close, close, volume)
    
    print(f"Input shape: {close.shape} (Time, Assets)")
    print(f"Output shape: {feats.shape} (Time-30, Assets, Features)")
    
    # Check for NaNs
    if np.isnan(feats).any():
        print("WARNING: Output contains NaNs!")
    else:
        print("SUCCESS: Features computed cleanly.")
