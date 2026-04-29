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
"""The core APIs of nvitop."""

from nvitop.api import (
    caching,
    collector,
    device,
    host,
    libamdsmi,
    libcuda,
    libcudart,
    libnvml,
    process,
    termcolor,
    utils,
)
from nvitop.api.caching import ttl_cache
from nvitop.api.collector import ResourceMetricCollector, collect_in_background, take_snapshots
from nvitop.api.amd_device import AmdPhysicalDevice
from nvitop.api.device import (
    CudaDevice,
    CudaMigDevice,
    Device,
    MigDevice,
    PhysicalDevice,
    normalize_cuda_visible_devices,
    parse_cuda_visible_devices,
)
from nvitop.api.libnvml import NVMLError, nvmlCheckReturn
from nvitop.api.process import GpuProcess, HostProcess, command_join
from nvitop.api.utils import (  # explicitly export these to appease mypy
    NA,
    SIZE_UNITS,
    UINT_MAX,
    ULONGLONG_MAX,
    GiB,
    KiB,
    MiB,
    NaType,
    NotApplicable,
    NotApplicableType,
    PiB,
    Snapshot,
    TiB,
    boolify,
    bytes2human,
    colored,
    human2bytes,
    set_color,
    timedelta2human,
    utilization2string,
)


def get_all_devices() -> list:
    """Return all GPU devices in the system — both NVIDIA and AMD.

    NVIDIA GPUs are returned first (indices 0..N-1), AMD GPUs follow
    (indices N..N+M-1).  Use this instead of :meth:`Device.all` when AMD
    GPU support is desired.

    Returns:
        A list of :class:`PhysicalDevice` and/or :class:`AmdPhysicalDevice`
        instances for every GPU detected on the system.
    """
    nvidia_devices = []
    try:
        nvidia_devices = Device.all()
    except libnvml.NVMLError:
        pass

    amd_devices = []
    try:
        nvidia_count = len(nvidia_devices)
        amd_devices = [
            AmdPhysicalDevice(index=i, unified_index=i + nvidia_count)
            for i in range(AmdPhysicalDevice.count())
        ]
    except Exception:  # noqa: BLE001
        pass

    return nvidia_devices + amd_devices


def get_device_count() -> int:
    """Return the total number of GPUs (NVIDIA + AMD) in the system."""
    nvidia_count = 0
    try:
        nvidia_count = Device.count()
    except libnvml.NVMLError:
        pass

    amd_count = 0
    try:
        amd_count = AmdPhysicalDevice.count()
    except Exception:  # noqa: BLE001
        pass

    return nvidia_count + amd_count


def get_driver_version() -> str:
    """Return the active GPU driver version string.

    Returns the NVIDIA display driver version on NVIDIA systems, the amdgpu
    driver version on AMD-only systems, or ``NA`` if neither is available.
    """
    try:
        version = Device.driver_version()
        if version is not NA:
            return version
    except libnvml.NVMLError:
        pass

    try:
        return AmdPhysicalDevice.driver_version()
    except Exception:  # noqa: BLE001
        pass

    return NA


def get_cuda_driver_version() -> str:
    """Return the CUDA or ROCm runtime version string.

    Returns the maximum CUDA version on NVIDIA systems, the ROCm version on
    AMD-only systems, or ``NA`` if neither is available.
    """
    try:
        version = Device.cuda_driver_version()
        if version is not NA:
            return version
    except libnvml.NVMLError:
        pass

    try:
        return AmdPhysicalDevice.rocm_version()
    except Exception:  # noqa: BLE001
        pass

    return NA


__all__ = [  # noqa: RUF022
    'NVMLError',
    'nvmlCheckReturn',
    'libnvml',
    'libcuda',
    'libcudart',
    # nvitop.api.amd_device
    'AmdPhysicalDevice',
    # nvitop.api (AMD-aware entry points)
    'get_all_devices',
    'get_device_count',
    'get_driver_version',
    'get_cuda_driver_version',
    # nvitop.api.device
    'Device',
    'PhysicalDevice',
    'MigDevice',
    'CudaDevice',
    'CudaMigDevice',
    'parse_cuda_visible_devices',
    'normalize_cuda_visible_devices',
    # nvitop.api.process
    'host',
    'HostProcess',
    'GpuProcess',
    'command_join',
    # nvitop.api.collector
    'take_snapshots',
    'collect_in_background',
    'ResourceMetricCollector',
    # nvitop.api.caching
    'ttl_cache',
    # nvitop.api.utils
    'NA',
    'NaType',
    'NotApplicable',
    'NotApplicableType',
    'UINT_MAX',
    'ULONGLONG_MAX',
    'KiB',
    'MiB',
    'GiB',
    'TiB',
    'PiB',
    'SIZE_UNITS',
    'bytes2human',
    'human2bytes',
    'timedelta2human',
    'utilization2string',
    'colored',
    'set_color',
    'boolify',
    'Snapshot',
]
