#!/usr/bin/env python
"""
Quick demo/test script for Numin meta-learning model.
This trains for a few epochs to verify everything is working.
"""

import torch
import sys
from src.dataloader import get_dataloaders_meta, get_dataloaders_standard
from src.model import MetaOHLCVPredictor

def main():
    print("=" * 80)
    print("Numin Meta-Learning - Quick Test")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Test meta-learning mode
    print("Testing Meta-Learning Mode...")
    print("-" * 80)
    
    try:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            get_dataloaders_meta(
                batch_size=4,
                lookback=10,
                k_shot=3,
                q_query=1,
                num_workers=0
            )
        
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Val samples: {len(val_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        
        # Get batch
        batch = next(iter(train_loader))
        print(f"✓ Batch shapes:")
        print(f"  - support_x: {batch['support_x'].shape}")
        print(f"  - support_y: {batch['support_y'].shape}")
        print(f"  - query_x: {batch['query_x'].shape}")
        print(f"  - query_y: {batch['query_y'].shape}")
        
        # Create model
        input_size = batch['support_x'].shape[-1]
        output_size = batch['support_y'].shape[-1]
        
        model = MetaOHLCVPredictor(
            input_size=input_size,
            dim=64,
            num_heads=4,
            output_size=output_size
        )
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {total_params:,}")
        
        # Forward pass
        support_x = batch['support_x'].to(device)
        support_y = batch['support_y'].to(device)
        query_x = batch['query_x'].to(device)
        query_y = batch['query_y'].to(device)
        
        with torch.no_grad():
            output = model(support_x, support_y, query_x)
        
        print(f"✓ Model output shape: {output.shape}")
        
        # Test loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, query_y)
        print(f"✓ Test loss: {loss.item():.6f}")
        
        print("\n✅ Meta-Learning mode: PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Meta-Learning mode: FAILED")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # Test standard mode
    print("Testing Standard Mode...")
    print("-" * 80)
    
    try:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            get_dataloaders_standard(
                batch_size=16,
                lookback=10,
                num_workers=0
            )
        
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Val samples: {len(val_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        
        # Get batch
        batch = next(iter(train_loader))
        print(f"✓ Batch shapes:")
        print(f"  - x: {batch['x'].shape}")
        print(f"  - y: {batch['y'].shape}")
        
        # Create model
        input_size = batch['x'].shape[-1]
        output_size = batch['y'].shape[-1]
        
        model = MetaOHLCVPredictor(
            input_size=input_size,
            dim=64,
            num_heads=4,
            output_size=output_size
        )
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {total_params:,}")
        
        # Forward pass
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        with torch.no_grad():
            output = model.forward_standard(x)
        
        print(f"✓ Model output shape: {output.shape}")
        
        # Test loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, y)
        print(f"✓ Test loss: {loss.item():.6f}")
        
        print("\n✅ Standard mode: PASSED\n")
        
    except Exception as e:
        print(f"\n❌ Standard mode: FAILED")
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nYou can now train with:")
    print("  python train.py --meta --epochs 10  (for meta-learning)")
    print("  python train.py --epochs 10         (for standard mode)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
