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
"""AMD GPU device implementation using AMD SMI library.

This module provides AmdPhysicalDevice which supports AMD GPUs alongside
the existing NVIDIA GPU support in nvitop.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from nvitop.api import libamdsmi
from nvitop.api.device import (
    ClockInfos,
    ClockSpeedInfos,
    Device,
    MemoryInfo,
    ThroughputInfo,
    UtilizationRates,
)
from nvitop.api.process import GpuProcess
from nvitop.api.utils import (
    NA,
    NaType,
    Snapshot,
    bytes2human,
    memoize_when_activated,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Hashable


__all__ = ['AmdPhysicalDevice']


class AmdPhysicalDevice(Device):
    """Device class for AMD GPUs, using AMD SMI (amdsmi) library.

    This class provides the same interface as the NVIDIA Device class but
    uses the AMD SMI library to query GPU information.
    """

    GPU_PROCESS_CLASS: ClassVar[type[GpuProcess]] = GpuProcess

    _amd_handles: ClassVar[list | None] = None
    _amd_handle_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        index: int | None = None,
        *,
        uuid: str | None = None,
        bus_id: str | None = None,
    ) -> None:
        """Initialize an AMD GPU device.

        Args:
            index: The AMD GPU index (0-based within AMD GPUs only).
            uuid: The UUID of the AMD GPU.
            bus_id: The PCI bus ID of the AMD GPU.
        """
        # Do NOT call super().__init__() because it tries to use NVML
        # Instead, initialize the same attributes manually

        self._name: str = NA
        self._uuid: str = NA
        self._bus_id: str = NA
        self._memory_total: int | NaType = NA
        self._memory_total_human: str = NA
        self._nvlink_link_count: int | None = None
        self._nvlink_throughput_counters: tuple[tuple[int | NaType, int]] | None = None
        self._is_mig_device: bool | None = False
        self._cuda_index: int | None = None
        self._cuda_compute_capability: tuple[int, int] | NaType | None = None

        self._handle = None  # Not used for AMD, but keep for compatibility

        # Initialize AMD SMI
        libamdsmi._lazy_init()

        # Get AMD GPU handles
        gpu_handles = self._get_amd_handles()

        self._amd_handle = None

        if index is not None:
            self._nvml_index = index
            if index < len(gpu_handles):
                self._amd_handle = gpu_handles[index]
            else:
                self._name = f'ERROR: AMD GPU index {index} out of range'
        elif uuid is not None:
            # Find GPU by UUID
            for i, h in enumerate(gpu_handles):
                try:
                    dev_uuid = libamdsmi.amdsmi_query(
                        'amdsmi_get_gpu_device_uuid',
                        h,
                        default=None,
                    )
                    if dev_uuid is not None and uuid.lower() in str(dev_uuid).lower():
                        self._amd_handle = h
                        self._nvml_index = i
                        break
                except Exception:  # noqa: BLE001
                    pass
            if self._amd_handle is None:
                self._nvml_index = NA  # type: ignore[assignment]
                self._name = f'ERROR: AMD GPU with UUID {uuid} not found'
        elif bus_id is not None:
            # Find GPU by bus ID
            bus_id_str = bus_id if isinstance(bus_id, str) else bus_id.decode()
            for i, h in enumerate(gpu_handles):
                try:
                    dev_bdf = libamdsmi.amdsmi_query(
                        'amdsmi_get_gpu_device_bdf',
                        h,
                        default=None,
                    )
                    if dev_bdf is not None and bus_id_str.lower() in str(dev_bdf).lower():
                        self._amd_handle = h
                        self._nvml_index = i
                        break
                except Exception:  # noqa: BLE001
                    pass
            if self._amd_handle is None:
                self._nvml_index = NA  # type: ignore[assignment]
                self._name = f'ERROR: AMD GPU with bus ID {bus_id} not found'

        self._max_clock_infos: ClockInfos = ClockInfos(graphics=NA, sm=NA, memory=NA, video=NA)
        self._lock: threading.RLock = threading.RLock()

        self._ident: tuple[Hashable, str] = (self.index, self.uuid())
        self._hash: int | None = None

        # AMD-specific cached data
        self._driver_version: str | NaType | None = None
        self._rocm_version: str | NaType | None = None

    @classmethod
    def _get_amd_handles(cls) -> list:
        """Get cached AMD GPU handles."""
        if cls._amd_handles is None:
            with cls._amd_handle_lock:
                if cls._amd_handles is None:
                    cls._amd_handles = libamdsmi.get_gpu_handles()
        return cls._amd_handles

    @classmethod
    def _refresh_handles(cls) -> None:
        """Refresh the cached AMD GPU handles."""
        with cls._amd_handle_lock:
            cls._amd_handles = None

    @classmethod
    def is_available(cls) -> bool:
        """Test whether there are any AMD devices and the AMD SMI library is loaded."""
        return libamdsmi.is_available()

    @staticmethod
    def driver_version() -> str | NaType:
        """The version of the AMD GPU driver."""
        try:
            libamdsmi._lazy_init()
            handles = libamdsmi.get_gpu_handles()
            if handles:
                driver_info = libamdsmi.amdsmi_query(
                    'amdsmi_get_gpu_driver_info',
                    handles[0],
                    default=None,
                )
                if driver_info is not None and isinstance(driver_info, dict):
                    version = driver_info.get('driver_version', NA)
                    if version and version != 'N/A':
                        return str(version)
            return NA
        except Exception:  # noqa: BLE001
            return NA

    @staticmethod
    def cuda_driver_version() -> str | NaType:
        """AMD GPUs use ROCm/HIP, not CUDA. Returns ROCm version if available."""
        return AmdPhysicalDevice.rocm_version()

    max_cuda_version = cuda_driver_version

    @staticmethod
    def rocm_version() -> str | NaType:
        """The ROCm version."""
        try:
            libamdsmi._lazy_init()
            amdsmi = libamdsmi.get_amdsmi_module()
            version = amdsmi.amdsmi_get_rocm_version()
            if isinstance(version, tuple):
                # Returns (success_bool, version_string) or just version_string
                if len(version) == 2 and isinstance(version[1], str):
                    return version[1]
                return str(version)
            if isinstance(version, dict):
                major = version.get('major', '')
                minor = version.get('minor', '')
                patch = version.get('patch', '')
                return f'{major}.{minor}.{patch}'
            return str(version)
        except Exception:  # noqa: BLE001
            return NA

    @staticmethod
    def cuda_runtime_version() -> str | NaType:
        """Not applicable for AMD GPUs."""
        return NA

    cudart_version = cuda_runtime_version

    @classmethod
    def count(cls) -> int:
        """The number of AMD GPUs in the system."""
        try:
            return len(cls._get_amd_handles())
        except Exception:  # noqa: BLE001
            return 0

    @classmethod
    def all(cls) -> list[AmdPhysicalDevice]:
        """Return a list of all AMD physical devices in the system."""
        return cls.from_indices()

    @classmethod
    def from_indices(cls, indices=None) -> list[AmdPhysicalDevice]:
        """Return a list of AMD devices of the given indices."""
        if indices is None:
            try:
                indices = range(cls.count())
            except Exception:  # noqa: BLE001
                return []

        if isinstance(indices, int):
            indices = [indices]

        devices = []
        for idx in indices:
            try:
                devices.append(cls(index=idx))
            except Exception:  # noqa: BLE001
                pass
        return devices

    def __new__(cls, *args, **kwargs):
        """Create a new instance directly as AmdPhysicalDevice."""
        return object.__new__(cls)

    def __repr__(self) -> str:
        """Return a string representation of the AMD device."""
        return '{}(index={}, name={!r}, total_memory={})'.format(  # noqa: UP032
            self.__class__.__name__,
            self.index,
            self.name(),
            self.memory_total_human(),
        )

    def __getattr__(self, name: str) -> Any:
        """Get the object attribute.

        For AMD devices, we don't do the dynamic NVML function resolution.
        Instead, return a function that returns NA for unknown attributes.
        """
        # Don't interfere with internal caching attributes
        if name.startswith('_'):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'",
            )

        try:
            return super(Device, self).__getattribute__(name)
        except AttributeError:
            # Return a callable that returns NA for unsupported NVML attributes
            def not_supported(*args: Any, **kwargs: Any) -> NaType:
                return NA

            not_supported.__name__ = name
            not_supported.__qualname__ = f'{self.__class__.__name__}.{name}'
            setattr(self, name, not_supported)
            return not_supported

    def __reduce__(self):
        """Return state information for pickling."""
        return self.__class__, (self._nvml_index,)

    @property
    def handle(self):
        """The AMD SMI handle (returns None as we don't use NVML handles)."""
        return None

    # ==================== Identity Methods ====================

    def name(self) -> str | NaType:
        """The product name of the AMD GPU."""
        if self._amd_handle is not None and self._name is NA:
            # Try board_info first for product name
            board_info = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_board_info',
                self._amd_handle,
                default=None,
            )
            if board_info is not None and isinstance(board_info, dict):
                product_name = board_info.get('product_name', None)
                if product_name and product_name != 'N/A':
                    self._name = str(product_name)
                    return self._name

            # Fallback to ASIC info
            asic_info = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_asic_info',
                self._amd_handle,
                default=None,
            )
            if asic_info is not None and isinstance(asic_info, dict):
                market_name = asic_info.get('market_name', None)
                if market_name and market_name != 'N/A':
                    self._name = str(market_name)
                    return self._name

        return self._name

    def uuid(self) -> str | NaType:
        """The UUID of the AMD GPU device."""
        if self._amd_handle is not None and self._uuid is NA:
            uuid_val = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_device_uuid',
                self._amd_handle,
                default=NA,
            )
            if uuid_val is not NA and uuid_val is not None:
                self._uuid = str(uuid_val)
        return self._uuid

    def bus_id(self) -> str | NaType:
        """PCI bus ID of the AMD GPU."""
        if self._amd_handle is not None and self._bus_id is NA:
            bdf = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_device_bdf',
                self._amd_handle,
                default=NA,
            )
            if bdf is not NA and bdf is not None:
                self._bus_id = str(bdf)
        return self._bus_id

    def serial(self) -> str | NaType:
        """The serial number of the AMD GPU."""
        if self._amd_handle is not None:
            asic_info = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_asic_info',
                self._amd_handle,
                default=None,
            )
            if asic_info is not None and isinstance(asic_info, dict):
                serial = asic_info.get('asic_serial', None)
                if serial and serial != 'N/A':
                    return str(serial)
        return NA

    # ==================== Memory Methods ====================

    @memoize_when_activated
    def memory_info(self) -> MemoryInfo:
        """Return memory information for the AMD GPU."""
        if self._amd_handle is not None:
            amdsmi = libamdsmi.get_amdsmi_module()
            try:
                total = amdsmi.amdsmi_get_gpu_memory_total(
                    self._amd_handle,
                    amdsmi.AmdSmiMemoryType.VRAM,
                )
                used = amdsmi.amdsmi_get_gpu_memory_usage(
                    self._amd_handle,
                    amdsmi.AmdSmiMemoryType.VRAM,
                )
                if isinstance(total, int) and isinstance(used, int):
                    free = total - used
                    return MemoryInfo(total=total, free=free, used=used, reserved=NA)
            except Exception:  # noqa: BLE001
                pass
        return MemoryInfo(total=NA, free=NA, used=NA, reserved=NA)

    @memoize_when_activated
    def bar1_memory_info(self) -> MemoryInfo:
        """BAR1 memory info - not directly available for AMD GPUs."""
        return MemoryInfo(total=NA, free=NA, used=NA)

    # ==================== Utilization Methods ====================

    @memoize_when_activated
    def utilization_rates(self) -> UtilizationRates:
        """Return GPU utilization rates for the AMD GPU."""
        gpu, memory, encoder, decoder = NA, NA, NA, NA

        if self._amd_handle is not None:
            activity = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_activity',
                self._amd_handle,
                default=None,
            )
            if activity is not None and isinstance(activity, dict):
                gfx = activity.get('gfx_activity', NA)
                if isinstance(gfx, int):
                    gpu = gfx
                umc = activity.get('umc_activity', NA)
                if isinstance(umc, int):
                    memory = umc
                mm = activity.get('mm_activity', NA)
                if isinstance(mm, int):
                    encoder = mm

            # Decoder is N/A for most AMD GPUs, use VCN activity if available
            try:
                amdsmi = libamdsmi.get_amdsmi_module()
                metrics = amdsmi.amdsmi_get_gpu_metrics_info(self._amd_handle)
                if isinstance(metrics, dict):
                    vcn = metrics.get('vcn_activity', NA)
                    if isinstance(vcn, (list, tuple)) and len(vcn) > 0:
                        # Sum VCN activity
                        vals = [v for v in vcn if isinstance(v, int)]
                        if vals:
                            decoder = max(vals)
                    elif isinstance(vcn, int):
                        decoder = vcn
            except Exception:  # noqa: BLE001
                pass

        return UtilizationRates(gpu=gpu, memory=memory, encoder=encoder, decoder=decoder)

    # ==================== Clock Methods ====================

    @memoize_when_activated
    def clock_infos(self) -> ClockInfos:
        """Return current clock speeds for the AMD GPU."""
        graphics, sm, memory, video = NA, NA, NA, NA

        if self._amd_handle is not None:
            amdsmi = libamdsmi.get_amdsmi_module()

            # GFX clock (graphics/compute)
            gfx_clock = libamdsmi.amdsmi_query(
                'amdsmi_get_clock_info',
                self._amd_handle,
                amdsmi.AmdSmiClkType.GFX,
                default=None,
            )
            if gfx_clock is not None and isinstance(gfx_clock, dict):
                clk = gfx_clock.get('clk', NA)
                if isinstance(clk, int):
                    graphics = clk
                    sm = clk  # SM equivalent for AMD

            # Memory clock
            mem_clock = libamdsmi.amdsmi_query(
                'amdsmi_get_clock_info',
                self._amd_handle,
                amdsmi.AmdSmiClkType.MEM,
                default=None,
            )
            if mem_clock is not None and isinstance(mem_clock, dict):
                clk = mem_clock.get('clk', NA)
                if isinstance(clk, int):
                    memory = clk

            # Video clock (VCLK0)
            try:
                vid_clock = libamdsmi.amdsmi_query(
                    'amdsmi_get_clock_info',
                    self._amd_handle,
                    amdsmi.AmdSmiClkType.VCLK0,
                    default=None,
                )
                if vid_clock is not None and isinstance(vid_clock, dict):
                    clk = vid_clock.get('clk', NA)
                    if isinstance(clk, int):
                        video = clk
            except Exception:  # noqa: BLE001
                pass

        return ClockInfos(graphics=graphics, sm=sm, memory=memory, video=video)

    clocks = clock_infos

    @memoize_when_activated
    def max_clock_infos(self) -> ClockInfos:
        """Return maximum clock speeds for the AMD GPU."""
        if self._amd_handle is not None:
            amdsmi = libamdsmi.get_amdsmi_module()

            graphics, sm, memory, video = NA, NA, NA, NA

            gfx_clock = libamdsmi.amdsmi_query(
                'amdsmi_get_clock_info',
                self._amd_handle,
                amdsmi.AmdSmiClkType.GFX,
                default=None,
            )
            if gfx_clock is not None and isinstance(gfx_clock, dict):
                max_clk = gfx_clock.get('max_clk', NA)
                if isinstance(max_clk, int):
                    graphics = max_clk
                    sm = max_clk

            mem_clock = libamdsmi.amdsmi_query(
                'amdsmi_get_clock_info',
                self._amd_handle,
                amdsmi.AmdSmiClkType.MEM,
                default=None,
            )
            if mem_clock is not None and isinstance(mem_clock, dict):
                max_clk = mem_clock.get('max_clk', NA)
                if isinstance(max_clk, int):
                    memory = max_clk

            try:
                vid_clock = libamdsmi.amdsmi_query(
                    'amdsmi_get_clock_info',
                    self._amd_handle,
                    amdsmi.AmdSmiClkType.VCLK0,
                    default=None,
                )
                if vid_clock is not None and isinstance(vid_clock, dict):
                    max_clk = vid_clock.get('max_clk', NA)
                    if isinstance(max_clk, int):
                        video = max_clk
            except Exception:  # noqa: BLE001
                pass

            self._max_clock_infos = ClockInfos(
                graphics=graphics,
                sm=sm,
                memory=memory,
                video=video,
            )

        return self._max_clock_infos

    max_clocks = max_clock_infos

    # ==================== Fan Speed ====================

    def fan_speed(self) -> int | NaType:
        """Fan speed as a percentage of maximum speed."""
        if self._amd_handle is not None:
            try:
                amdsmi = libamdsmi.get_amdsmi_module()
                speed = amdsmi.amdsmi_get_gpu_fan_speed(self._amd_handle, 0)
                max_speed = amdsmi.amdsmi_get_gpu_fan_speed_max(self._amd_handle, 0)
                if isinstance(speed, int) and isinstance(max_speed, int) and max_speed > 0:
                    return round(100 * speed / max_speed)
            except Exception:  # noqa: BLE001
                # Fan speed not supported on this device (e.g., MI300A)
                pass

            # Try metrics for fan speed
            try:
                amdsmi = libamdsmi.get_amdsmi_module()
                metrics = amdsmi.amdsmi_get_gpu_metrics_info(self._amd_handle)
                if isinstance(metrics, dict):
                    fan = metrics.get('current_fan_speed', NA)
                    if isinstance(fan, int):
                        # Fan speed in metrics is raw RPM or percentage
                        return fan
            except Exception:  # noqa: BLE001
                pass
        return NA

    # ==================== Temperature ====================

    def temperature(self) -> int | NaType:
        """GPU temperature in degrees Celsius."""
        if self._amd_handle is not None:
            amdsmi = libamdsmi.get_amdsmi_module()

            # Try hotspot temperature first (most equivalent to NVIDIA's GPU temp)
            for temp_type in [
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureType.JUNCTION,
                amdsmi.AmdSmiTemperatureType.EDGE,
            ]:
                try:
                    temp = amdsmi.amdsmi_get_temp_metric(
                        self._amd_handle,
                        temp_type,
                        amdsmi.AmdSmiTemperatureMetric.CURRENT,
                    )
                    if isinstance(temp, int):
                        return temp
                except Exception:  # noqa: BLE001
                    continue
        return NA

    # ==================== Power ====================

    @memoize_when_activated
    def power_usage(self) -> int | NaType:
        """Power draw in milliwatts."""
        if self._amd_handle is not None:
            power_info = libamdsmi.amdsmi_query(
                'amdsmi_get_power_info',
                self._amd_handle,
                default=None,
            )
            if power_info is not None and isinstance(power_info, dict):
                # Try socket_power first (current), then current_socket_power
                for key in ['current_socket_power', 'socket_power', 'average_socket_power']:
                    power = power_info.get(key, None)
                    if isinstance(power, (int, float)) and power > 0:
                        # amdsmi reports power in watts, convert to milliwatts
                        return int(power * 1000)
        return NA

    power_draw = power_usage

    @memoize_when_activated
    def power_limit(self) -> int | NaType:
        """Power limit in milliwatts."""
        if self._amd_handle is not None:
            power_cap = libamdsmi.amdsmi_query(
                'amdsmi_get_power_cap_info',
                self._amd_handle,
                default=None,
            )
            if power_cap is not None and isinstance(power_cap, dict):
                cap = power_cap.get('power_cap', None)
                if isinstance(cap, int) and cap > 0:
                    # amdsmi reports power cap in microwatts, convert to milliwatts
                    return cap // 1000
        return NA

    # ==================== PCIe Throughput ====================

    @memoize_when_activated
    def pcie_tx_throughput(self) -> int | NaType:
        """PCIe transmit throughput in KiB/s."""
        # PCIe throughput is not easily available for all AMD GPUs
        return NA

    @memoize_when_activated
    def pcie_rx_throughput(self) -> int | NaType:
        """PCIe receive throughput in KiB/s."""
        return NA

    # ==================== Display / Driver / Mode ====================

    def display_active(self) -> str | NaType:
        """Display active status. Not directly available for AMD."""
        return 'N/A'

    def display_mode(self) -> str | NaType:
        """Display mode. Not directly available for AMD."""
        return 'N/A'

    def current_driver_model(self) -> str | NaType:
        """Driver model. Not applicable for AMD on Linux."""
        return 'N/A'

    driver_model = current_driver_model

    def persistence_mode(self) -> str | NaType:
        """Persistence mode. Not applicable for AMD GPUs."""
        return 'N/A'

    def performance_state(self) -> str | NaType:
        """Performance state (PowerPlay level) for AMD GPU."""
        if self._amd_handle is not None:
            perf_level = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_perf_level',
                self._amd_handle,
                default=NA,
            )
            if perf_level is not NA and perf_level is not None:
                perf_str = str(perf_level)
                # Map AMD perf levels to short names
                if 'AUTO' in perf_str:
                    return 'Auto'
                if 'HIGH' in perf_str:
                    return 'High'
                if 'LOW' in perf_str:
                    return 'Low'
                if 'MANUAL' in perf_str:
                    return 'Man'
                return perf_str.replace('AMDSMI_DEV_PERF_LEVEL_', '')
        return NA

    def total_volatile_uncorrected_ecc_errors(self) -> int | NaType:
        """Total uncorrected ECC errors."""
        if self._amd_handle is not None:
            ecc = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_total_ecc_count',
                self._amd_handle,
                default=None,
            )
            if ecc is not None and isinstance(ecc, dict):
                uncorr = ecc.get('uncorrectable_count', NA)
                if isinstance(uncorr, int):
                    return uncorr
        return NA

    def compute_mode(self) -> str | NaType:
        """Compute mode for AMD GPU."""
        if self._amd_handle is not None:
            # AMD uses compute partition instead of compute mode
            partition = libamdsmi.amdsmi_query(
                'amdsmi_get_gpu_compute_partition',
                self._amd_handle,
                default=None,
            )
            if partition is not None:
                part_str = str(partition)
                if 'SPX' in part_str:
                    return 'SPX'
                if 'DPX' in part_str:
                    return 'DPX'
                if 'TPX' in part_str:
                    return 'TPX'
                if 'QPX' in part_str:
                    return 'QPX'
                if 'CPX' in part_str:
                    return 'CPX'
                return str(partition)
        return 'Default'

    def cuda_compute_capability(self) -> tuple[int, int] | NaType:
        """CUDA compute capability. Not applicable for AMD GPUs."""
        return NA

    def is_mig_device(self) -> bool:
        """AMD GPUs don't support MIG in the NVIDIA sense."""
        return False

    def mig_mode(self) -> str | NaType:
        """MIG mode. Not applicable for AMD GPUs."""
        return 'N/A'

    def is_mig_mode_enabled(self) -> bool:
        """MIG mode not applicable for AMD."""
        return False

    def max_mig_device_count(self) -> int:
        """No MIG support for AMD."""
        return 0

    def mig_devices(self) -> list:
        """No MIG devices for AMD."""
        return []

    # ==================== Process Monitoring ====================

    def processes(self) -> dict[int, GpuProcess]:
        """Return a dictionary of processes running on the AMD GPU.

        Returns:
            Dict[int, GpuProcess]: A dictionary mapping PID to GPU process instance.
        """
        if self._amd_handle is None:
            return {}

        processes = {}
        amdsmi = libamdsmi.get_amdsmi_module()

        # Method 1: Use amdsmi_get_gpu_process_list for per-device processes
        try:
            proc_list = amdsmi.amdsmi_get_gpu_process_list(self._amd_handle)
            for proc_info in proc_list:
                if isinstance(proc_info, dict):
                    pid = proc_info.get('pid', proc_info.get('process_id', None))
                    if pid is None:
                        continue
                    gpu_memory = proc_info.get('vram_usage', NA)
                    if not isinstance(gpu_memory, int):
                        gpu_memory = NA
                else:
                    # proc_info might be a handle - try to get info from it
                    try:
                        info = amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(proc_info)
                        pid = info.get('process_id', info.get('pid', None))
                        if pid is None:
                            continue
                        gpu_memory = info.get('vram_usage', NA)
                        if not isinstance(gpu_memory, int):
                            gpu_memory = NA
                    except Exception:  # noqa: BLE001
                        continue

                proc = self.GPU_PROCESS_CLASS(
                    pid=pid,
                    device=self,
                    gpu_memory=gpu_memory,
                    gpu_instance_id=0xFFFFFFFF,
                    compute_instance_id=0xFFFFFFFF,
                )
                proc.type = proc.type + 'C'  # Mark as compute process
                processes[pid] = proc
        except Exception:  # noqa: BLE001
            pass

        # Method 2: Fallback to system-wide compute process info
        if len(processes) == 0:
            try:
                compute_procs = amdsmi.amdsmi_get_gpu_compute_process_info()
                for proc_info in compute_procs:
                    if isinstance(proc_info, dict):
                        pid = proc_info.get('process_id', proc_info.get('pid', None))
                        if pid is None:
                            continue
                        gpu_memory = proc_info.get('vram_usage', NA)
                        if not isinstance(gpu_memory, int):
                            gpu_memory = NA

                        proc = self.GPU_PROCESS_CLASS(
                            pid=pid,
                            device=self,
                            gpu_memory=gpu_memory,
                            gpu_instance_id=0xFFFFFFFF,
                            compute_instance_id=0xFFFFFFFF,
                        )
                        proc.type = proc.type + 'C'
                        processes[pid] = proc
            except Exception:  # noqa: BLE001
                pass

        return processes

    # ==================== Snapshot ====================

    def as_snapshot(self) -> Snapshot:
        """Return a one-time snapshot of the AMD device."""
        with self.oneshot():
            return Snapshot(
                real=self,
                index=self.index,
                physical_index=self.physical_index,
                **{key: getattr(self, key)() for key in self.SNAPSHOT_KEYS},
            )

    SNAPSHOT_KEYS: ClassVar[list[str]] = [
        'name',
        'uuid',
        'bus_id',
        'memory_info',
        'memory_used',
        'memory_free',
        'memory_total',
        'memory_used_human',
        'memory_free_human',
        'memory_total_human',
        'memory_percent',
        'memory_usage',
        'utilization_rates',
        'gpu_utilization',
        'memory_utilization',
        'encoder_utilization',
        'decoder_utilization',
        'clock_infos',
        'max_clock_infos',
        'clock_speed_infos',
        'sm_clock',
        'memory_clock',
        'fan_speed',
        'temperature',
        'power_usage',
        'power_limit',
        'power_status',
        'pcie_throughput',
        'pcie_tx_throughput',
        'pcie_rx_throughput',
        'pcie_tx_throughput_human',
        'pcie_rx_throughput_human',
        'display_active',
        'display_mode',
        'current_driver_model',
        'persistence_mode',
        'performance_state',
        'total_volatile_uncorrected_ecc_errors',
        'compute_mode',
        'cuda_compute_capability',
        'mig_mode',
    ]

    @contextlib.contextmanager
    def oneshot(self) -> Generator[None]:
        """Context manager to batch-query multiple device metrics."""
        with self._lock:
            if hasattr(self, '_cache'):
                yield
            else:
                try:
                    self.memory_info.cache_activate(self)  # type: ignore[attr-defined]
                    self.bar1_memory_info.cache_activate(self)  # type: ignore[attr-defined]
                    self.utilization_rates.cache_activate(self)  # type: ignore[attr-defined]
                    self.clock_infos.cache_activate(self)  # type: ignore[attr-defined]
                    self.max_clock_infos.cache_activate(self)  # type: ignore[attr-defined]
                    self.power_usage.cache_activate(self)  # type: ignore[attr-defined]
                    self.power_limit.cache_activate(self)  # type: ignore[attr-defined]
                    yield
                finally:
                    self.memory_info.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.bar1_memory_info.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.utilization_rates.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.clock_infos.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.max_clock_infos.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.power_usage.cache_deactivate(self)  # type: ignore[attr-defined]
                    self.power_limit.cache_deactivate(self)  # type: ignore[attr-defined]

    # ==================== Inherited helper methods ====================
    # These are inherited from Device and work correctly since they call
    # the methods we've overridden above:
    # - memory_total(), memory_used(), memory_free()
    # - memory_total_human(), memory_used_human(), memory_free_human()
    # - memory_percent(), memory_usage()
    # - gpu_utilization(), memory_utilization()
    # - encoder_utilization(), decoder_utilization()
    # - clock_speed_infos(), graphics_clock(), sm_clock(), memory_clock(), video_clock()
    # - max_graphics_clock(), max_sm_clock(), max_memory_clock(), max_video_clock()
    # - power_status()
    # - pcie_throughput(), pcie_tx_throughput_human(), pcie_rx_throughput_human()
    # - is_leaf_device(), to_leaf_devices()

    # NVLink methods return empty/NA for AMD
    def nvlink_link_count(self) -> int:
        """Number of XGMI links for AMD GPU."""
        return 0

    def nvlink_throughput(self, interval=None) -> list[ThroughputInfo]:
        """XGMI throughput - not directly available via this method."""
        return []
