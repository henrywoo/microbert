#!/usr/bin/env python3
"""
Script to check PyTorch compatibility and available features
"""

import sys
import torch

def check_pytorch_compatibility():
    """Check PyTorch version and available features"""
    
    print("=" * 60)
    print("PYTORCH COMPATIBILITY CHECK")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: ✓ Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("CUDA available: ✗ No")
    
    # Check specific scheduler availability
    print("\nLearning Rate Scheduler Compatibility:")
    
    # Check for get_linear_schedule_with_warmup
    try:
        from transformers import get_linear_schedule_with_warmup
        print("  get_linear_schedule_with_warmup: ✓ Available (from transformers)")
    except ImportError:
        print("  get_linear_schedule_with_warmup: ✗ Not available (transformers not installed)")
    
    # Check for other schedulers
    available_schedulers = []
    scheduler_classes = [
        'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
        'ReduceLROnPlateau', 'OneCycleLR', 'CosineAnnealingWarmRestarts'
    ]
    
    for scheduler_name in scheduler_classes:
        try:
            getattr(torch.optim.lr_scheduler, scheduler_name)
            available_schedulers.append(scheduler_name)
        except AttributeError:
            pass
    
    print(f"  Other available schedulers: {', '.join(available_schedulers)}")
    
    # Check for mixed precision training
    print("\nMixed Precision Training:")
    try:
        autocast = torch.cuda.amp.autocast
        print("  torch.cuda.amp.autocast: ✓ Available")
    except AttributeError:
        print("  torch.cuda.amp.autocast: ✗ Not available")
    
    try:
        grad_scaler = torch.cuda.amp.GradScaler
        print("  torch.cuda.amp.GradScaler: ✓ Available")
    except AttributeError:
        print("  torch.cuda.amp.GradScaler: ✗ Not available")
    
    # Check for bfloat16 support
    print("\nData Type Support:")
    try:
        bfloat16 = torch.bfloat16
        print("  torch.bfloat16: ✓ Available")
    except AttributeError:
        print("  torch.bfloat16: ✗ Not available (will use float16)")
    
    try:
        float16 = torch.float16
        print("  torch.float16: ✓ Available")
    except AttributeError:
        print("  torch.float16: ✗ Not available")
    
    # Check for distributed training
    print("\nDistributed Training:")
    try:
        distributed = torch.distributed
        print("  torch.distributed: ✓ Available")
    except AttributeError:
        print("  torch.distributed: ✗ Not available")
    
    # Check for gradient checkpointing
    print("\nMemory Optimization:")
    try:
        checkpointing = torch.utils.checkpoint
        print("  torch.utils.checkpoint: ✓ Available")
    except AttributeError:
        print("  torch.utils.checkpoint: ✗ Not available")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✓ CUDA is available - GPU training should work")
    else:
        print("⚠ CUDA not available - training will be slow on CPU")
    
    # Check if we need fallback scheduler
    try:
        from transformers import get_linear_schedule_with_warmup
        print("✓ Linear schedule with warmup available (from transformers)")
    except ImportError:
        print("⚠ Transformers not available - need to install for get_linear_schedule_with_warmup")
    
    # Check mixed precision
    try:
        torch.cuda.amp.autocast
        torch.cuda.amp.GradScaler
        print("✓ Mixed precision training available")
    except AttributeError:
        print("⚠ Mixed precision not available - training may use more memory")
    
    # Check bfloat16
    try:
        torch.bfloat16
        print("✓ bfloat16 available (good for H200/A100)")
    except AttributeError:
        print("⚠ bfloat16 not available - will use float16")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✅ PyTorch setup looks good for GPU training")
        print("   The training script should work with automatic fallbacks")
    else:
        print("⚠️  PyTorch setup limited - CPU training only")
        print("   Consider installing CUDA version of PyTorch for better performance")

def main():
    try:
        check_pytorch_compatibility()
    except Exception as e:
        print(f"Error during compatibility check: {e}")
        print("This might indicate a PyTorch installation issue")

if __name__ == '__main__':
    main()
