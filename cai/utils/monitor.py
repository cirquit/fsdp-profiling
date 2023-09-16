import os
import time
import psutil

import threading
import subprocess as sp
import time
from pynvml import *
    #nvmlInit, \
    #nvmlDeviceGetCount, \
    #nvmlDeviceGetHandleByIndex, \
    #nvmlSystemGetDriverVersion, \
    #nvmlDeviceGetName, \
    #nvmlDeviceGetPciInfo, \
    #nvmlDeviceGetVbiosVersion, \
    #nvmlDeviceGetMaxPcieLinkGeneration, \
    #nvmlDeviceGetCurrPcieLinkGeneration, \
    #nvmlDeviceGetMaxPcieLinkWidth, \
    #nvmlDeviceGetCurrPcieLinkWidth
#import wandb

#def start_monitor(monitor_log_frequency_ms):
#    def inner():
#        mon = Monitor()
#        wandb.log(mon.get_static_info())
#        monitor_log_frequency_s = monitor_log_frequency_ms / 1000
#        while True:
#            wandb.log({**mon.get_sys_info()})
#            time.sleep(monitor_log_frequency_s)

#    t = threading.Thread(target=inner)
#    t.daemon = True
#    t.start()

class Monitor:
    def __init__(self, cuda_enabled=False):

        # 
        self.cuda_enabled=cuda_enabled
        self.gpus_attached=0
        self.gpu_handles=[]
        if self.cuda_enabled:
            nvmlInit()
            self.gpus_attached = nvmlDeviceGetCount()
            for ix in range(self.gpus_attached):
                self.gpu_handles.append(nvmlDeviceGetHandleByIndex(ix))

        # setting up t-1 values for bandwidth calculation
        self.disk_read_sys_mb, self.disk_write_sys_mb = 0, 0
        self.net_sent_sys_mbit, self.net_recv_sys_mbit = 0, 0
        self.bandwidth_snapshot_time_s = 0

        # setting up t=0 values for global system increasing values
        (
            self.global_ctx_switches_count_init,
            self.global_interrupts_count_init,
            self.global_soft_interrupts_count_init,
        ) = (0, 0, 0)
        self.disk_read_sys_count_init, self.disk_write_sys_count_init = 0, 0
        self.disk_read_time_sys_s_init, self.disk_write_time_sys_s_init = 0, 0
        self.disk_busy_time_sys_s_init = 0

        # populating the snapshot values
        self.create_bandwidth_snapshot()
        self.create_interrupt_snapshot()
        self.create_disk_access_snapshot()

    def __exit(self, exc_type, exc_value, traceback):
        if self.cuda_enabled:
            nvmlShutdown()

    def get_static_info(self):
        logical_core_count = psutil.cpu_count(logical=True)
        total_b, _, _, _, _, _, _, _, _, _, _ = psutil.virtual_memory()
        total_memory_sys_MB = total_b / 1000**2

        cuda_dict = {}
        if self.cuda_enabled:
            cuda_dict["gpu_count"] = str(nvmlDeviceGetCount())
            cuda_dict["driver_version"] = str(nvmlSystemGetDriverVersion())
            for ix, handle in enumerate(self.gpu_handles):
                cuda_dict[f"gpu_{ix}_name"] = \
                    str(nvmlDeviceGetName(handle))
                cuda_dict[f"gpu_{ix}_id"] = \
                    str(nvmlDeviceGetPciInfo(handle).busId)
                cuda_dict[f"gpu_{ix}_vbios_version"] = \
                    str(nvmlDeviceGetVbiosVersion(handle))
                cuda_dict[f"gpu_{ix}_pci_max_link"] = \
                    str(nvmlDeviceGetMaxPcieLinkGeneration(handle))
                cuda_dict[f"gpu_{ix}_pci_cur_link"] = \
                    str(nvmlDeviceGetCurrPcieLinkGeneration(handle))
                cuda_dict[f"gpu_{ix}_pci_max_link_width"] = \
                    str(nvmlDeviceGetMaxPcieLinkWidth(handle))
                cuda_dict[f"gpu_{ix}_pci_cur_link_width"] = \
                    str(nvmlDeviceGetCurrPcieLinkWidth(handle))

        return {
            **cuda_dict,
            "cpu_core_count": logical_core_count,
            "total_memory_sys_MB": total_memory_sys_MB,
        }

    def get_sys_info(self):
        """Get the current system and process info of the Python runtime.
        Bandwidths are calculated over the last interval since this method was called
        """
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()
        proc_info = self.get_process_info()
        disk_info = self.get_disk_info()
        net_info = self.get_network_info()
        bandwidth_info = self.get_bandwidths(disk_info=disk_info, net_info=net_info)
        if self.cuda_enabled:
            gpu_mem_info = self.get_cuda_memory_info()
            gpu_smi_info = self.get_nvidia_smi_info()
        else:
            gpu_mem_info = {}
            gpu_smi_info = {}

        # remove the global counters
        del disk_info["disk/disk_read_sys_MB"]
        del disk_info["disk/disk_write_sys_MB"]
        del net_info["network/net_sent_sys_mbit"]
        del net_info["network/net_recv_sys_mbit"]

        return {
            **cpu_info,
            **memory_info,
            **proc_info,
            **disk_info,
            **net_info,
            **bandwidth_info,
            **gpu_mem_info,
            **gpu_smi_info
        }

    def create_disk_access_snapshot(self):
        """Sets the disk counters to the initial value which is subtracted in `get_disk_info` to get a "per-run" count"""
        disk_info = self.get_disk_info()
        self.disk_read_sys_count_init = disk_info["disk/counter/disk_read_sys_count"]
        self.disk_write_sys_count_init = disk_info["disk/counter/disk_write_sys_count"]
        self.disk_read_time_sys_s_init = disk_info["disk/time/disk_read_time_sys_s"]
        self.disk_write_time_sys_s_init = disk_info["disk/time/disk_write_time_sys_s"]
        self.disk_busy_time_sys_s_init = disk_info["disk/time/disk_busy_time_sys_s"]

    def create_interrupt_snapshot(self):
        """Sets the interrupt counters to the initial value which is subtracted in `get_cpu_info` to get a "per-run" count"""
        cpu_info = self.get_cpu_info()
        self.global_ctx_switches_count_init = cpu_info["cpu/interrupts/ctx_switches_count"]
        self.global_interrupts_count_init = cpu_info["cpu/interrupts/interrupts_count"]
        self.global_soft_interrupts_count_init = cpu_info["cpu/interrupts/soft_interrupts_count"]

    def create_bandwidth_snapshot(self, disk_info=None, net_info=None):
        """Sets the disk and network counters + time to calculate the bandwidth on the next call of `get_bandwidths`"""
        if disk_info == None:
            disk_info = self.get_disk_info()
        self.disk_read_sys_mb = disk_info["disk/disk_read_sys_MB"]
        self.disk_write_sys_mb = disk_info["disk/disk_write_sys_MB"]

        if net_info == None:
            net_info = self.get_network_info()
        self.net_sent_sys_mbit = net_info["network/net_sent_sys_mbit"]
        self.net_recv_sys_mbit = net_info["network/net_recv_sys_mbit"]

        self.bandwidth_snapshot_s = time.time()

    def get_bandwidths(self, disk_info, net_info):
        """Calculate the difference between the disk and network read/written bytes since the last call
        Populates the member variables that cached the last state + time
        """
        # todo: use a deque with size 2
        old_disk_read_sys_mb = self.disk_read_sys_mb
        old_disk_write_sys_mb = self.disk_write_sys_mb
        old_net_sent_sys_mbit = self.net_sent_sys_mbit
        old_net_recv_sys_mbit = self.net_recv_sys_mbit
        old_bandwidth_snapshot_s = self.bandwidth_snapshot_s

        self.create_bandwidth_snapshot()

        disk_read_sys_timeframe_mb = self.disk_read_sys_mb - old_disk_read_sys_mb
        disk_write_sys_timeframe_mb = self.disk_write_sys_mb - old_disk_write_sys_mb
        net_sent_sys_timeframe_mbit = self.net_sent_sys_mbit - old_net_sent_sys_mbit
        net_recv_sys_timeframe_mbit = self.net_recv_sys_mbit - old_net_recv_sys_mbit
        time_diff_s = self.bandwidth_snapshot_s - old_bandwidth_snapshot_s

        disk_read_sys_bandwidth_mbs = disk_read_sys_timeframe_mb / time_diff_s
        disk_write_sys_bandwidth_mbs = disk_write_sys_timeframe_mb / time_diff_s
        net_sent_sys_bandwidth_mbs = net_sent_sys_timeframe_mbit / time_diff_s
        net_recv_sys_bandwidth_mbs = net_recv_sys_timeframe_mbit / time_diff_s

        return {
            "bandwidth/disk_read_sys_bandwidth_MBs": disk_read_sys_bandwidth_mbs,
            "bandwidth/disk_write_sys_bandwidth_MBs": disk_write_sys_bandwidth_mbs,
            "bandwidth/net_sent_sys_bandwidth_Mbits": net_sent_sys_bandwidth_mbs,
            "bandwidth/net_recv_sys_bandwidth_Mbits": net_recv_sys_bandwidth_mbs,
        }

    def get_cpu_info(self):
        # hyperthreaded cores included
        # type: int
        logical_core_count = psutil.cpu_count(logical=True)

        # global cpu stats, ever increasing from boot, we null them for easier comparison from each init of this class
        # type: (int, int, int, int)
        (
            global_ctx_switches_count,
            global_interrupts_count,
            global_soft_interrupts_count,
            _,
        ) = psutil.cpu_stats()
        ctx_switches_count = global_ctx_switches_count - self.global_ctx_switches_count_init
        interrupts_count = global_interrupts_count - self.global_interrupts_count_init
        soft_interrupts_count = global_soft_interrupts_count - self.global_soft_interrupts_count_init

        # average system load over 1, 5 and 15 minutes summarized over all cores in percent
        # type: (float, float, float)
        one_min, five_min, fifteen_min = psutil.getloadavg()
        avg_sys_load_one_min_percent = one_min / logical_core_count * 100
        avg_sys_load_five_min_percent = five_min / logical_core_count * 100
        avg_sys_load_fifteen_min_percent = fifteen_min / logical_core_count * 100

        return {
            "cpu/interrupts/ctx_switches_count": ctx_switches_count,
            "cpu/interrupts/interrupts_count": interrupts_count,
            "cpu/interrupts/soft_interrupts_count": soft_interrupts_count,
            "cpu/load/avg_sys_load_one_min_percent": avg_sys_load_one_min_percent,
            "cpu/load/avg_sys_load_five_min_percent": avg_sys_load_five_min_percent,
            "cpu/load/avg_sys_load_fifteen_min_percent": avg_sys_load_fifteen_min_percent,
        }

    @staticmethod
    def get_memory_info():
        # global memory information
        # type (int): total_b - total memory on the system in bytes
        # type (int): available_b - available memory on the system in bytes
        # type (float): used_percent - total / used_b
        # type (int): used_b - used memory on the system in bytes (may not match "total - available" or "total - free")
        (
            total_b,
            available_b,
            used_memory_sys_percent,
            used_b,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = psutil.virtual_memory()
        available_memory_sys_mb = available_b / 1000**2
        used_memory_sys_mb = used_b / 1000**2

        return {
            "memory/available_memory_sys_MB": available_memory_sys_mb,
            "memory/used_memory_sys_MB": used_memory_sys_mb,
            "memory/used_memory_sys_percent": used_memory_sys_percent,
        }

    @staticmethod
    def get_process_info():

        # gets its own pid by default
        proc = psutil.Process()

        # voluntary and involunatry context switches by the process (cumulative)
        # type: (int, int)
        (
            voluntary_proc_ctx_switches,
            involuntary_proc_ctx_switches,
        ) = proc.num_ctx_switches()

        # memory information
        # type (int): rrs_b - resident set size: non-swappable physical memory used in bytes
        # type (int): vms_b - virtual memory size: total amount of virtual memory used in bytes
        # type (int): shared_b - shared memory size in bytes
        # type (int): trs_b - text resident set: memory devoted to executable code in bytes
        # type (int): drs_b - data resident set: physical memory devoted to other than code in bytes
        # type (int): lib_b - library memory: memory used by shared libraries in bytes
        # type (int): dirty_pages_count - number of dirty pages
        (
            rss_b,
            vms_b,
            shared_b,
            trs_b,
            drs_b,
            lib_b,
            dirty_pages_proc_count,
        ) = proc.memory_info()
        resident_set_size_proc_mb = rss_b / 1000**2
        virtual_memory_size_proc_mb = vms_b / 1000**2
        shared_memory_proc_mb = shared_b / 1000**2
        text_resident_set_proc_mb = trs_b / 1000**2
        data_resident_set_proc_mb = drs_b / 1000**2
        lib_memory_proc_mb = lib_b / 1000**2

        return {
            "process/voluntary_proc_ctx_switches": voluntary_proc_ctx_switches,
            "process/involuntary_proc_ctx_switches": involuntary_proc_ctx_switches,
            "process/memory/resident_set_size_proc_MB": resident_set_size_proc_mb,
            "process/memory/virtual_memory_size_proc_MB": virtual_memory_size_proc_mb,
            "process/memory/shared_memory_proc_MB": shared_memory_proc_mb,
            "process/memory/text_resident_set_proc_MB": text_resident_set_proc_mb,
            "process/memory/data_resident_set_proc_MB": data_resident_set_proc_mb,
            "process/memory/lib_memory_proc_MB": lib_memory_proc_mb,
            "process/memory/dirty_pages_proc_count": dirty_pages_proc_count,
        }

    def get_disk_info(self):

        # system disk stats
        # type (int): disk_read_sys_count - how often were reads performed
        # type (int): disk_write_sys_count - how often were writes performed
        # type (int): disk_read_sys_bytes - how much was read in bytes
        # type (int): writen_sys_bytes - how much was written in bytes
        # type (int): disk_read_time_sys_ms - how much time was used to read in milliseconds
        # type (int): disk_write_time_sys_ms - how much time was used to write in milliseconds
        # type (int): busy_time_sys_ms - how much time was used for actual I/O

        (
            global_disk_read_sys_count,
            global_disk_write_sys_count,
            global_disk_read_sys_bytes,
            global_disk_write_sys_bytes,
            global_disk_read_time_sys_ms,
            global_disk_write_time_sys_ms,
            _,
            _,
            global_busy_time_sys_ms,
        ) = psutil.disk_io_counters()

        disk_read_sys_mb = global_disk_read_sys_bytes / 1000**2
        disk_write_sys_mb = global_disk_write_sys_bytes / 1000**2
        # subtracting global system start to get process values
        disk_read_sys_count = global_disk_read_sys_count - self.disk_read_sys_count_init
        disk_write_sys_count = global_disk_write_sys_count - self.disk_write_sys_count_init
        disk_read_time_sys_s = global_disk_read_time_sys_ms / 1000 - self.disk_read_time_sys_s_init
        disk_write_time_sys_s = global_disk_write_time_sys_ms / 1000 - self.disk_write_time_sys_s_init
        disk_busy_time_sys_s = global_busy_time_sys_ms / 1000 - self.disk_busy_time_sys_s_init

        return {
            "disk/counter/disk_read_sys_count": disk_read_sys_count,
            "disk/counter/disk_write_sys_count": disk_write_sys_count,
            "disk/disk_read_sys_MB": disk_read_sys_mb,
            "disk/disk_write_sys_MB": disk_write_sys_mb,
            "disk/time/disk_read_time_sys_s": disk_read_time_sys_s,
            "disk/time/disk_write_time_sys_s": disk_write_time_sys_s,
            "disk/time/disk_busy_time_sys_s": disk_busy_time_sys_s,
        }

    @staticmethod
    def get_network_info():

        # network system stats
        # type (int): net_sent_sys_bytes - sent bytes over all network interfaces
        # type (int): net_recv_sys_bytes - received bytes over all network interfaces
        (
            net_sent_sys_bytes,
            net_recv_sys_bytes,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = psutil.net_io_counters(pernic=False)
        net_sent_sys_mbit = (net_sent_sys_bytes / 1000**2) * 8
        net_recv_sys_mbit = (net_recv_sys_bytes / 1000**2) * 8

        return {
            "network/net_sent_sys_mbit": net_sent_sys_mbit,
            "network/net_recv_sys_mbit": net_recv_sys_mbit,
        }

    def get_nvidia_smi_info(self):
        return_dict = {}

        for ix, handle in enumerate(self.gpu_handles):
            group_key = f"05_gpu_smi/gpu_{ix}_"

            # https://man.archlinux.org/man/nvidia-smi.1.en
            # actual memory on the device
            frameBufferMemInfo = nvmlDeviceGetMemoryInfo(handle)
            fb_mem_total_MiB = frameBufferMemInfo.total / 1024 / 1024
            fb_mem_used_MiB = frameBufferMemInfo.used / 1024 / 1024
            fb_mem_free_MiB = fb_mem_total_MiB - fb_mem_used_MiB
            return_dict[group_key + "fb_total_MiB"] = fb_mem_total_MiB
            return_dict[group_key + "fb_used_MiB"] = fb_mem_used_MiB
            return_dict[group_key + "fb_free_MiB"] = fb_mem_free_MiB

            # mapped FB memory to be accessed by the CPU via P2P over the PCIe bus
            bar1MemInfo = nvmlDeviceGetBAR1MemoryInfo(handle)
            bar1_mem_total_MiB = bar1MemInfo.bar1Total / 1024 / 1024
            bar1_mem_used_MiB = bar1MemInfo.bar1Used / 1024 / 1024
            bar1_mem_free_MiB = bar1_mem_total_MiB - bar1_mem_used_MiB
            return_dict[group_key + "bar1_total_MiB"] = bar1_mem_total_MiB
            return_dict[group_key + "bar1_used_MiB"] = bar1_mem_used_MiB
            return_dict[group_key + "bar1_free_MiB"] = bar1_mem_free_MiB

            # device utilization
            utilization = nvmlDeviceGetUtilizationRates(handle)
            return_dict[group_key + "gpu_util_in_percent"] = utilization.gpu
            return_dict[group_key + "mem_util_in_percent"] = utilization.memory

            # device temperature
            temp_in_c = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            return_dict[group_key + "temp_in_C"] = temp_in_c

            # https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html
            # throttle reasons, logged in a binary fashion to plot them against each other
            throttleReasons = [
                [nvmlClocksThrottleReasonGpuIdle, \
                    "clocks_throttle_reason_gpu_idle"],
                [nvmlClocksThrottleReasonUserDefinedClocks, \
                    "clocks_throttle_reason_user_defined_clocks"],
                [nvmlClocksThrottleReasonApplicationsClocksSetting, \
                    "clocks_throttle_reason_applications_clocks_setting"],
                [nvmlClocksThrottleReasonSwPowerCap, \
                    "clocks_throttle_reason_sw_power_cap"],
                [nvmlClocksThrottleReasonHwSlowdown, \
                    "clocks_throttle_reason_hw_slowdown"],
            ];
            supportedClocksThrottleReasons = \
                nvmlDeviceGetSupportedClocksThrottleReasons(handle);
            clocksThrottleReasons = nvmlDeviceGetCurrentClocksThrottleReasons(handle);
            for (mask, name) in throttleReasons:
                if (name != "clocks_throttle_reason_user_defined_clocks"):
                    if (mask & supportedClocksThrottleReasons):
                        val = 1 if mask & clocksThrottleReasons else 0;
                    return_dict[group_key + name] = val

        return {
            **return_dict
        }



    @staticmethod
    def get_cuda_memory_info():
        # cuda memory stats
        # https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats
        import torch
        gpu_key = "03_gpu_mem/"
        memory_summary_dict = torch.cuda.memory_stats(device=torch.cuda.current_device())
        return_dict = {}
        for key in memory_summary_dict.keys():
            return_key = gpu_key + key
            value = memory_summary_dict[key]
            return_dict[return_key] = value

        # memory fragmentation is computed via the memory consumption of all allocated segments
        # divided by the total size of all segments, as this can apparently be different?
        # source: https://github.com/pytorch/pytorch/issues/29554#issuecomment-668745542
        # 0.8 - 1.0: allocated size matches total size => not much fragmentation?
        # 0.0 - 0.2: allocated size is much smaller than total size => a lot fragmentation?
        snapshot = torch.cuda.memory_snapshot()
        return_dict["04_agg_gpu/memory_fragmentation_ratio"] = \
            sum(b['allocated_size'] for b in snapshot) / \
            sum(b['total_size'] for b in snapshot)

        # average segment size (cudaMalloc() calls vs its size)
        # if high: consecutive memory allocated => good
        # if low: a lot of non-consecutive memory allocated => bad
        #return_dict["04_agg_gpu/avg_segment_size_current_bytes"] = \
        #    memory_summary_dict["requested_bytes.all.current"] / \
        #    memory_summary_dict["segment.all.current"]
        
        # large pool > 1MB
        #return_dict["04_agg_gpu/avg_large_segment_size_current_bytes"] = \
        #    memory_summary_dict["requested_bytes.large_pool.current"] / \
        #    memory_summary_dict["segment.large_pool.current"]
        # small pool < 1MB
        #return_dict["04_agg_gpu/avg_small_segment_size_current_bytes"] = \
        #    memory_summary_dict["requested_bytes.small_pool.current"] / \
        #    memory_summary_dict["segment.small_pool.current"]

        # average active block size (inner parts of a segment that was allocated by cudaMalloc())
        # if high: good
        # if low: bad
        return_dict["04_agg_gpu/avg_block_size_current_bytes"] = \
            memory_summary_dict["active_bytes.all.current"] / \
            memory_summary_dict["active.all.current"]

        # average inactive block size (inactive + non-releasable)
        # if high: good as its easier to remove with torch.cuda.empty_cache?
        # if low: bad?
        return_dict["04_agg_gpu/avg_inactive_block_size_current_bytes"] = \
            memory_summary_dict["inactive_split_bytes.all.current"] / \
            memory_summary_dict["inactive_split.all.current"]

        # caching allocator rounds requests to minimize memory fragmentation
        # the different between allocated - requested bytes is the overhead that
        # comes from rounding
        # if high: bad, as we're wasting memory to reduce fragmentation
        # if log: good, we're not wasting memory
        #return_dict["04_agg_gpu/allocation_rounding_overhead_bytes"] = \
        #    memory_summary_dict["allocated_bytes.all.current"] - \
        #    memory_summary_dict["requested_bytes.all.current"]

        return {
            **return_dict
        }


