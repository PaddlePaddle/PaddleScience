import os
import subprocess
import numpy as np
import paddle

from paddle_geometric.graphgym.config import cfg


def get_gpu_memory_map():
    """Get the current GPU usage."""
    try:
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
        return gpu_memory
    except subprocess.CalledProcessError as e:
        print(f"Error accessing GPU memory: {e}")
        return np.array([])


def get_current_gpu_usage():
    """Get the current GPU memory usage for the current process."""
    if cfg.gpu_mem and cfg.device != 'cpu' and paddle.device.is_compiled_with_cuda():
        try:
            result = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
            current_pid = os.getpid()
            used_memory = 0
            for line in result.strip().split('\n'):
                line = line.split(', ')
                if current_pid == int(line[0]):
                    used_memory += int(line[1])
            return used_memory
        except subprocess.CalledProcessError as e:
            print(f"Error accessing GPU memory usage: {e}")
            return -1
    else:
        return -1


def auto_select_device():
    """Automatically select device for the current experiment."""
    if cfg.accelerator == 'auto':
        if paddle.device.is_compiled_with_cuda():
            cfg.accelerator = 'gpu'
            cfg.devices = 1
        else:
            cfg.accelerator = 'cpu'
            cfg.devices = None
