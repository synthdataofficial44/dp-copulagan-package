"""
Helper utilities for DP-CopulaGAN package.

This module provides utility functions for:
- Random seed management (reproducibility)
- GPU configuration
- Device information
- Logging setup
"""

import os
import random
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - TensorFlow
    - PYTHONHASHSEED environment variable
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    
    Examples
    --------
    >>> from dp_copulagan.utils import set_random_seed
    >>> set_random_seed(42)
    >>> # All subsequent operations will be deterministic
    
    Notes
    -----
    For full reproducibility, this should be called at the start of your script.
    Some TensorFlow operations may still be non-deterministic on GPU due to
    hardware-level parallelism.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Configure TensorFlow for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"✓ Random seed set to {seed} (deterministic mode enabled)")


def setup_gpu(memory_growth: bool = True, gpu_id: Optional[int] = None):
    """
    Configure GPU settings for TensorFlow.
    
    This function:
    - Enables memory growth to prevent OOM errors
    - Optionally selects a specific GPU
    - Returns GPU availability status
    
    Parameters
    ----------
    memory_growth : bool, default=True
        Enable dynamic memory allocation (recommended).
    gpu_id : Optional[int], default=None
        Specific GPU ID to use. If None, uses all available GPUs.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise.
    
    Examples
    --------
    >>> from dp_copulagan.utils import setup_gpu
    >>> gpu_available = setup_gpu(memory_growth=True)
    >>> if gpu_available:
    ...     print("Training on GPU")
    ... else:
    ...     print("Training on CPU")
    
    Notes
    -----
    Memory growth prevents TensorFlow from allocating all GPU memory at once,
    which allows multiple processes to share the GPU.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  No GPU detected. Training will use CPU (slower).")
        return False
    
    try:
        if gpu_id is not None:
            # Use specific GPU
            if gpu_id >= len(gpus):
                raise ValueError(f"GPU {gpu_id} not available. Found {len(gpus)} GPU(s).")
            gpus = [gpus[gpu_id]]
            tf.config.set_visible_devices(gpus, 'GPU')
            print(f"✓ Using GPU {gpu_id}: {gpus[0].name}")
        else:
            print(f"✓ Found {len(gpus)} GPU(s)")
        
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled")
        
        return True
    
    except RuntimeError as e:
        print(f"⚠️  GPU configuration failed: {e}")
        return False


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information for logging and reproducibility.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'gpu_available': bool
        - 'num_gpus': int
        - 'gpu_names': list of str
        - 'gpu_memory': list of memory capacities (if available)
        - 'cpu_count': int
    
    Examples
    --------
    >>> from dp_copulagan.utils import get_device_info
    >>> info = get_device_info()
    >>> print(f"Training on {info['num_gpus']} GPU(s)")
    """
    import multiprocessing
    
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'gpu_available': len(gpus) > 0,
        'num_gpus': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
        'cpu_count': multiprocessing.cpu_count(),
    }
    
    # Try to get GPU memory info
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            memory_values = [int(x) for x in result.stdout.strip().split('\n')]
            info['gpu_memory_mb'] = memory_values
    except:
        info['gpu_memory_mb'] = None
    
    return info


def print_device_info():
    """
    Print human-readable device information.
    
    Examples
    --------
    >>> from dp_copulagan.utils import print_device_info
    >>> print_device_info()
    """
    info = get_device_info()
    
    print("="*80)
    print("Device Information")
    print("="*80)
    print(f"CPUs:          {info['cpu_count']} cores")
    print(f"GPUs:          {info['num_gpus']} device(s)")
    
    if info['gpu_available']:
        for i, name in enumerate(info['gpu_names']):
            mem_info = ""
            if info.get('gpu_memory_mb'):
                mem_gb = info['gpu_memory_mb'][i] / 1024
                mem_info = f" ({mem_gb:.1f} GB)"
            print(f"  GPU {i}:      {name}{mem_info}")
    else:
        print("  (No GPU available - using CPU)")
    
    print("="*80)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Parameters
    ----------
    seconds : float
        Time in seconds.
    
    Returns
    -------
    str
        Formatted time string (e.g., "1h 23m 45s").
    
    Examples
    --------
    >>> format_time(3665)
    '1h 1m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def ensure_dir(path: str):
    """
    Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str
        Directory path to create.
    
    Examples
    --------
    >>> ensure_dir('results/synthetic_data')
    """
    os.makedirs(path, exist_ok=True)


class ProgressPrinter:
    """
    Simple progress printer for training loops.
    
    Examples
    --------
    >>> printer = ProgressPrinter(total=1000, prefix="Training")
    >>> for epoch in range(1000):
    ...     # training code
    ...     printer.update(epoch, {'loss': 0.5, 'accuracy': 0.9})
    """
    
    def __init__(self, total: int, prefix: str = "Progress", print_every: int = 100):
        self.total = total
        self.prefix = prefix
        self.print_every = print_every
    
    def update(self, current: int, metrics: Optional[Dict[str, float]] = None):
        """Update progress and print if needed."""
        if (current + 1) % self.print_every == 0 or current == self.total - 1:
            progress = (current + 1) / self.total * 100
            msg = f"{self.prefix} {current+1:4d}/{self.total} ({progress:5.1f}%)"
            
            if metrics:
                metric_str = " │ ".join([f"{k}: {v:7.4f}" for k, v in metrics.items()])
                msg += f" │ {metric_str}"
            
            print(msg)
