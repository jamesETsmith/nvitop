# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper module for AMD SMI library (amdsmi).

This module provides a unified interface for querying AMD GPU information,
analogous to libnvml.py for NVIDIA GPUs.
"""

from __future__ import annotations

import threading
from typing import Any

from nvitop.api.utils import NA, NaType


__all__ = [
    'AmdSmiError',
    'amdsmi_init',
    'amdsmi_shutdown',
    'amdsmi_query',
    'is_available',
    'check_return',
    'lazy_init',
    'amd_device_count',
    'get_amdsmi_module',
    'get_gpu_handles',
    'get_gpu_count',
]


# Try to import amdsmi
_amdsmi = None
_amdsmi_available = False
_amdsmi_initialized = False
_init_lock = threading.Lock()
_init_refcount = 0


class AmdSmiError(Exception):
    """Base error class for AMD SMI operations."""


class AmdSmiError_LibraryNotFound(AmdSmiError):
    """AMD SMI library not found."""


class AmdSmiError_NotSupported(AmdSmiError):
    """Feature not supported on this device."""


class AmdSmiError_DriverNotLoaded(AmdSmiError):
    """AMD GPU driver not loaded."""


def _load_amdsmi():
    """Attempt to load the amdsmi Python module."""
    global _amdsmi, _amdsmi_available  # noqa: PLW0603
    try:
        import amdsmi  # noqa: PLC0415

        _amdsmi = amdsmi
        _amdsmi_available = True
    except ImportError:
        _amdsmi_available = False
    except Exception:  # noqa: BLE001
        _amdsmi_available = False


def _lazy_init():
    """Thread-safe lazy initialization of AMD SMI."""
    global _amdsmi_initialized, _init_refcount  # noqa: PLW0603

    if _amdsmi_initialized:
        return

    with _init_lock:
        if _amdsmi_initialized:
            return

        if _amdsmi is None:
            _load_amdsmi()

        if not _amdsmi_available:
            raise AmdSmiError_LibraryNotFound(
                'AMD SMI library (amdsmi) not found. '
                'Please install ROCm SMI or the amdsmi Python package.',
            )

        try:
            _amdsmi.amdsmi_init()
            _amdsmi_initialized = True
            _init_refcount = 1
        except Exception as e:
            raise AmdSmiError_DriverNotLoaded(
                f'Failed to initialize AMD SMI: {e}',
            ) from e


def amdsmi_init():
    """Initialize the AMD SMI library."""
    global _init_refcount  # noqa: PLW0603
    _lazy_init()
    with _init_lock:
        _init_refcount += 1


def amdsmi_shutdown():
    """Shutdown the AMD SMI library."""
    global _amdsmi_initialized, _init_refcount  # noqa: PLW0603
    with _init_lock:
        _init_refcount -= 1
        if _init_refcount <= 0 and _amdsmi_initialized:
            try:
                _amdsmi.amdsmi_shut_down()
            except Exception:  # noqa: BLE001
                pass
            _amdsmi_initialized = False
            _init_refcount = 0


def amdsmi_query(func, *args, default=NA, ignore_errors=True):
    """Call an AMD SMI function with error handling.

    Similar to libnvml.nvmlQuery, this wraps AMD SMI calls with
    proper error handling and returns NA on failure.

    Args:
        func: Either a callable or a string name of an amdsmi function.
        *args: Arguments to pass to the function.
        default: Default value to return on error (default: NA).
        ignore_errors: If True, return default on error instead of raising.

    Returns:
        The result of the function call, or default on error.
    """
    _lazy_init()

    if isinstance(func, str):
        func = getattr(_amdsmi, func)

    try:
        result = func(*args)
        return result
    except Exception as e:  # noqa: BLE001
        if not ignore_errors:
            raise AmdSmiError(str(e)) from e
        return default


def is_available() -> bool:
    """Check if AMD SMI is available and there are AMD GPUs."""
    try:
        _lazy_init()
        handles = get_gpu_handles()
        return len(handles) > 0
    except (AmdSmiError, Exception):  # noqa: BLE001
        return False


def get_gpu_handles() -> list:
    """Get handles for all AMD GPU processors."""
    _lazy_init()
    try:
        all_handles = _amdsmi.amdsmi_get_processor_handles()
        gpu_handles = []
        for h in all_handles:
            try:
                ptype = _amdsmi.amdsmi_get_processor_type(h)
                ptype_str = str(ptype.get('processor_type', ''))
                if 'AMD_GPU' in ptype_str:
                    gpu_handles.append(h)
            except Exception:  # noqa: BLE001
                pass
        return gpu_handles
    except Exception:  # noqa: BLE001
        return []


def get_gpu_count() -> int:
    """Get the number of AMD GPUs in the system."""
    return len(get_gpu_handles())


def check_return(value, expected_type=None) -> bool:
    """Check if a return value is valid (not NA).

    Args:
        value: The value to check.
        expected_type: Optional type to validate against.

    Returns:
        True if the value is valid, False otherwise.
    """
    if isinstance(value, NaType):
        return False
    if value is None:
        return False
    if isinstance(value, str) and value == 'N/A':
        return False
    if expected_type is not None:
        return isinstance(value, expected_type)
    return True


# Expose the amdsmi module for direct access if needed
def get_amdsmi_module():
    """Get the underlying amdsmi module."""
    _lazy_init()
    return _amdsmi


def lazy_init() -> None:
    """Public alias for thread-safe lazy initialization of AMD SMI."""
    _lazy_init()


def amd_device_count() -> int:
    """Return the number of AMD GPUs, or 0 if AMD SMI is unavailable."""
    return get_gpu_count()


# Temperature types
TEMP_TYPE_EDGE = 'EDGE'
TEMP_TYPE_HOTSPOT = 'HOTSPOT'
TEMP_TYPE_JUNCTION = 'JUNCTION'
TEMP_TYPE_VRAM = 'VRAM'

# Memory types
MEM_TYPE_VRAM = 'VRAM'
MEM_TYPE_VIS_VRAM = 'VIS_VRAM'
MEM_TYPE_GTT = 'GTT'

# Clock types
CLK_TYPE_GFX = 'GFX'
CLK_TYPE_SYS = 'SYS'
CLK_TYPE_MEM = 'MEM'
CLK_TYPE_SOC = 'SOC'
