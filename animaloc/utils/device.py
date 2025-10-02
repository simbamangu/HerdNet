"""
Device detection and management utilities for cross-platform PyTorch support.
Supports CUDA, MPS (Apple Silicon), and CPU devices.
"""

import torch
from typing import Optional, Union


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Auto-detect or use specified device.

    Args:
        device_name: Optional device name ('cuda', 'mps', 'cpu').
                    If None, automatically detects best available device.

    Returns:
        torch.device: The selected device.

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
        >>> device = get_device('mps')  # Force MPS
    """
    if device_name:
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.

    Returns:
        bool: True if CUDA or MPS is available.
    """
    return torch.cuda.is_available() or (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    )


def get_device_type() -> str:
    """
    Get the type of the best available device.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    return get_device().type


def setup_cudnn(benchmark: bool = True, deterministic: bool = False) -> None:
    """
    Setup cuDNN settings if CUDA is available.

    Args:
        benchmark: Enable cuDNN benchmark mode for performance.
        deterministic: Enable deterministic mode for reproducibility.

    Note:
        These settings only apply to CUDA. MPS uses different backends.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


def get_memory_allocated(device: Optional[torch.device] = None) -> float:
    """
    Get memory allocated on the device in MB.

    Args:
        device: Device to check. If None, uses current default device.

    Returns:
        float: Memory allocated in MB, or 0 if not available.
    """
    if device is None:
        device = get_device()

    MB = 1024.0 * 1024.0

    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / MB
    elif device.type == 'mps':
        # MPS memory tracking available in PyTorch 2.0+
        if hasattr(torch.mps, 'current_allocated_memory'):
            return torch.mps.current_allocated_memory() / MB
        return 0.0
    else:
        return 0.0


def get_max_memory_allocated(device: Optional[torch.device] = None) -> float:
    """
    Get maximum memory allocated on the device in MB.

    Args:
        device: Device to check. If None, uses current default device.

    Returns:
        float: Maximum memory allocated in MB, or 0 if not available.
    """
    if device is None:
        device = get_device()

    MB = 1024.0 * 1024.0

    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / MB
    elif device.type == 'mps':
        # MPS doesn't have max_memory_allocated yet
        # Fall back to current allocated
        if hasattr(torch.mps, 'current_allocated_memory'):
            return torch.mps.current_allocated_memory() / MB
        return 0.0
    else:
        return 0.0


def empty_cache(device: Optional[torch.device] = None) -> None:
    """
    Empty the device cache.

    Args:
        device: Device to clear cache for. If None, uses current default device.
    """
    if device is None:
        device = get_device()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def supports_pinned_memory(device: Optional[torch.device] = None) -> bool:
    """
    Check if the device supports pinned memory.

    Args:
        device: Device to check. If None, uses current default device.

    Returns:
        bool: True if pinned memory is supported.
    """
    if device is None:
        device = get_device()

    return device.type in ['cuda', 'mps']


def get_autocast_context(device: torch.device, enabled: bool = True):
    """
    Get the appropriate autocast context for the device.

    Args:
        device: The device to use for autocast.
        enabled: Whether to enable autocast.

    Returns:
        Context manager for automatic mixed precision.

    Examples:
        >>> device = get_device()
        >>> with get_autocast_context(device):
        ...     output = model(input)
    """
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        # PyTorch 2.0+ with device-agnostic autocast
        if device.type in ['cuda', 'mps']:
            return torch.amp.autocast(device_type=device.type, enabled=enabled)

    # Fallback for older PyTorch versions
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(enabled=enabled)

    # For CPU or if autocast not available, use a no-op context
    from contextlib import nullcontext
    return nullcontext()
