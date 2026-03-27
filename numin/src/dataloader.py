import torch
from torch.utils.data import DataLoader


def get_dataloaders_standard(
    batch_size=32,
    lookback=10,
    num_workers=0
):
    """
    Create standard supervised learning DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    from .dataset import OHLCVReturnsDataset
    
    train_dataset = OHLCVReturnsDataset(
        "data",
        lookback=lookback,
        split='train',
        meta_learning=False
    )
    val_dataset = OHLCVReturnsDataset(
        "data",
        lookback=lookback,
        split='val',
        meta_learning=False
    )
    test_dataset = OHLCVReturnsDataset(
        "data",
        lookback=lookback,
        split='test',
        meta_learning=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def get_dataloaders_meta(
    batch_size=16,
    lookback=10,
    num_workers=0,
    k_shot=5,
    q_query=1
):
    """
    Create meta-learning DataLoaders.
    
    Args:
        batch_size: Number of tasks per batch
        lookback: Sequence length
        num_workers: Number of workers
        k_shot: Support samples per task
        q_query: Query samples per task
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    from .dataset import OHLCVReturnsDataset
    
    train_dataset = OHLCVReturnsDataset(
        "data",
        lookback=lookback,
        split='train',
        meta_learning=True,
        k_shot=k_shot,
        q_query=q_query
    )
    val_dataset = OHLCVReturnsDataset(
        "data",
        lookback=lookback,
        split='val',
        meta_learning=True,
        k_shot=k_shot,
        q_query=q_query
    )
    test_dataset = OHLCVReturnsDataset(
        "data",
        lookback=lookback,
        split='test',
        meta_learning=True,
        k_shot=k_shot,
        q_query=q_query
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
