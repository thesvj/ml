import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class OHLCVReturnsDataset(Dataset):
    """
    Dataset for OHLCV-to-Returns prediction task with meta-learning setup.
    
    Can be used in two ways:
    1. Standard supervised learning: (input_sequence, target_returns)
    2. Few-shot meta-learning: Multiple tasks with support/query sets
    
    Args:
        data_dir: Path to directory containing parquet files
        lookback: Number of days to look back (sequence length)
        split: 'train', 'val', or 'test'
        meta_learning: If True, format data for meta-learning tasks
        n_way: Number of different assets/instruments (for meta-learning)
        k_shot: Number of support samples per task (for meta-learning)
        q_query: Number of query samples per task (for meta-learning)
    """
    
    def __init__(
        self,
        data_dir,
        lookback=10,
        split='train',
        test_size=0.1,
        val_size=0.2,
        meta_learning=False,
        n_way=5,
        k_shot=5,
        q_query=1
    ):
        self.lookback = lookback
        self.split = split
        self.meta_learning = meta_learning
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        data_dir = Path(data_dir)
        
        # Read parquet files
        ohlcv_df = pd.read_parquet(data_dir / "consolidated_daily_ohlcv.parquet")
        returns_df = pd.read_parquet(data_dir / "consolidated_daily_returns.parquet")
        
        # Ensure they have the same length
        min_len = min(len(ohlcv_df), len(returns_df))
        ohlcv_df = ohlcv_df.iloc[:min_len]
        returns_df = returns_df.iloc[:min_len]
        
        # Reset index
        ohlcv_df = ohlcv_df.reset_index(drop=True)
        returns_df = returns_df.reset_index(drop=True)
        
        self.ohlcv_df = ohlcv_df
        self.returns_df = returns_df
        self.n_assets = ohlcv_df.shape[1]
        
        # Split data
        n_samples = len(ohlcv_df)
        n_test = int(n_samples * test_size)
        n_val = int((n_samples - n_test) * val_size)
        
        if split == 'test':
            start_idx = n_samples - n_test
            end_idx = n_samples
        elif split == 'val':
            start_idx = n_samples - n_test - n_val
            end_idx = n_samples - n_test
        else:  # train
            start_idx = 0
            end_idx = n_samples - n_test - n_val
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        # Normalize OHLCV data
        self.scaler_ohlcv = StandardScaler()
        ohlcv_scaled = self.scaler_ohlcv.fit_transform(ohlcv_df.values)
        self.ohlcv_scaled = torch.from_numpy(ohlcv_scaled).float()
        
        # Normalize returns data
        self.scaler_returns = StandardScaler()
        returns_scaled = self.scaler_returns.fit_transform(returns_df.values)
        self.returns_scaled = torch.from_numpy(returns_scaled).float()
        
        # Calculate valid indices (must have lookback history)
        self.valid_indices = []
        for i in range(start_idx + lookback, end_idx):
            self.valid_indices.append(i)
    
    def __len__(self):
        if self.meta_learning:
            return len(self.valid_indices) // (self.k_shot + self.q_query)
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            Standard mode:
                x: (lookback, num_features) - OHLCV data for lookback days
                y: (num_features,) - Returns for the next day
            
            Meta-learning mode:
                support_x: (k_shot, lookback, num_features)
                support_y: (k_shot, num_features)
                query_x: (q_query, lookback, num_features)
                query_y: (q_query, num_features)
        """
        if self.meta_learning:
            return self._get_meta_task(idx)
        else:
            return self._get_standard_sample(idx)
    
    def _get_standard_sample(self, idx):
        i = self.valid_indices[idx]
        
        # Get lookback window of OHLCV data
        x = self.ohlcv_scaled[i - self.lookback:i]
        
        # Get target returns for day i
        y = self.returns_scaled[i]
        
        return {
            'x': x,
            'y': y,
            'index': i
        }
    
    def _get_meta_task(self, idx):
        """
        Create a meta-learning task with support and query sets.
        Both support and query use the same time window but different asset features.
        """
        base_idx = idx * (self.k_shot + self.q_query) + self.start_idx + self.lookback
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        # Create task with k_shot support samples and q_query query samples
        for j in range(self.k_shot):
            if base_idx + j < self.end_idx:
                i = base_idx + j
                x = self.ohlcv_scaled[i - self.lookback:i]
                y = self.returns_scaled[i]
                support_x.append(x)
                support_y.append(y)
        
        for j in range(self.q_query):
            if base_idx + self.k_shot + j < self.end_idx:
                i = base_idx + self.k_shot + j
                x = self.ohlcv_scaled[i - self.lookback:i]
                y = self.returns_scaled[i]
                query_x.append(x)
                query_y.append(y)
        
        # Pad if needed
        while len(support_x) < self.k_shot:
            support_x.append(torch.zeros_like(support_x[0]))
            support_y.append(torch.zeros_like(support_y[0]))
        
        while len(query_x) < self.q_query:
            query_x.append(torch.zeros_like(query_x[0]))
            query_y.append(torch.zeros_like(query_y[0]))
        
        return {
            'support_x': torch.stack(support_x),
            'support_y': torch.stack(support_y),
            'query_x': torch.stack(query_x),
            'query_y': torch.stack(query_y),
        }


if __name__ == "__main__":
    # Test standard dataset
    dataset_train = OHLCVReturnsDataset("data", lookback=10, split='train', meta_learning=False)
    print(f"Train samples (standard): {len(dataset_train)}")
    sample = dataset_train[0]
    print(f"Sample x shape: {sample['x'].shape}, y shape: {sample['y'].shape}")
    
    # Test meta-learning dataset
    meta_dataset_train = OHLCVReturnsDataset("data", lookback=10, split='train', meta_learning=True)
    print(f"\nTrain tasks (meta-learning): {len(meta_dataset_train)}")
    meta_sample = meta_dataset_train[0]
    print(f"Support x shape: {meta_sample['support_x'].shape}")
    print(f"Support y shape: {meta_sample['support_y'].shape}")
    print(f"Query x shape: {meta_sample['query_x'].shape}")
    print(f"Query y shape: {meta_sample['query_y'].shape}")
