import os
import pathlib
import time
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import paddle
from paddle.profiler import Profiler

from paddle_geometric.profile.utils import (
    byte_to_megabyte,
    get_gpu_memory_from_ipex,
    get_gpu_memory_from_nvidia_smi,
)

ProfilerActivity =None

@dataclass
class GPUStats:
    time: float
    max_allocated_gpu: float
    max_reserved_gpu: float
    max_active_gpu: float


@dataclass
class CUDAStats(GPUStats):
    nvidia_smi_free_cuda: float
    nvidia_smi_used_cuda: float


@dataclass
class GPUStatsSummary:
    time_mean: float
    time_std: float
    max_allocated_gpu: float
    max_reserved_gpu: float
    max_active_gpu: float


@dataclass
class CUDAStatsSummary(GPUStatsSummary):
    min_nvidia_smi_free_cuda: float
    max_nvidia_smi_used_cuda: float

def profileit(device: str):  # pragma: no cover
    r"""A decorator to facilitate profiling a function, *e.g.*, obtaining
    training runtime and memory statistics of a specific model on a specific
    dataset.
    Returns a :obj:`GPUStats` if :obj:`device` is :obj:`xpu` or extended
    object :obj:`CUDAStats`, if :obj:`device` is :obj:`cuda`.

    Args:
        device (str): Target device for profiling. Options are:
            :obj:`cuda` and obj:`xpu`.

    .. code-block:: python

        @profileit("cuda")
        def train(model, optimizer, x, edge_index, y):
            optimizer.clear_grad()
            out = model(x, edge_index)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            return float(loss)

        loss, stats = train(model, x, edge_index, y)
    """
    def decorator(func):
        def wrapper(
                *args, **kwargs
        ) -> Union[Tuple[Any, GPUStats], Tuple[Any, CUDAStats]]:
            model = args[0]
            if not isinstance(model, paddle.nn.Layer):
                raise AttributeError(
                    'First argument for profiling needs to be paddle.nn.Layer')
            if device not in ['cuda', 'xpu']:
                raise AttributeError(
                    "The profiling decorator supports only CUDA and "
                    "XPU devices")

            device_id = None
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, paddle.Tensor):
                    device_id = arg.place.gpu_device_id()
                    break
            if device_id is None:
                raise AttributeError(
                    "Could not infer GPU device from the args in the "
                    "function being profiled")
            if device_id == -1:
                raise RuntimeError(
                    "The profiling decorator does not support profiling "
                    "on non GPU devices")

            is_cuda = device == 'cuda'

            if is_cuda:
                from paddle.profiler import Profiler

                # Init Paddle profiler:
                profiler = Profiler(scheduler=paddle.profiler.make_scheduler(1, 1))
                profiler.start()

            start_event = paddle.device.cuda.Event(enable_timing=True)
            end_event = paddle.device.cuda.Event(enable_timing=True)
            start_event.record()

            out = func(*args, **kwargs)

            end_event.record()
            paddle.device.cuda.synchronize()
            time = paddle.device.cuda.elapsed_time(start_event, end_event) / 1000

            if is_cuda:
                profiler.stop()
                profiler.export('profiler_output')

                # Memory stats:
                mem_stats = paddle.device.cuda.memory_stats(device_id)
                max_allocated = mem_stats['allocated_bytes.all.peak']
                max_reserved = mem_stats['reserved_bytes.all.peak']
                max_active = mem_stats['active.all.peak']

                free_cuda, used_cuda = get_gpu_memory_from_nvidia_smi(device=device_id)

                stats = CUDAStats(time, max_allocated, max_reserved,
                                  max_active, free_cuda, used_cuda)
                return out, stats
            else:
                stats = GPUStats(time, *get_gpu_memory_from_ipex(device_id))
                return out, stats

        return wrapper

    return decorator



class timeit(ContextDecorator):
    r"""A context decorator to facilitate timing a function, *e.g.*, obtaining
    the runtime of a specific model on a specific dataset.

    .. code-block:: python

        @paddle.no_grad()
        def test(model, x, edge_index):
            return model(x, edge_index)

        with timeit() as t:
            z = test(model, x, edge_index)
        time = t.duration

    Args:
        log (bool, optional): If set to :obj:`False`, will not log any runtime
            to the console. (default: :obj:`True`)
        avg_time_divisor (int, optional): If set to a value greater than
            :obj:`1`, will divide the total time by this value. Useful for
            calculating the average of runtimes within a for-loop.
            (default: :obj:`0`)
    """
    def __init__(self, log: bool = True, avg_time_divisor: int = 0):
        self.log = log
        self.avg_time_divisor = avg_time_divisor

    def __enter__(self):
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        self.t_start = time.time()
        return self

    def __exit__(self, *args):
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        self.t_end = time.time()
        self.duration = self.t_end - self.t_start
        if self.avg_time_divisor > 1:
            self.duration = self.duration / self.avg_time_divisor
        if self.log:  # pragma: no cover
            print(f'Time: {self.duration:.8f}s', flush=True)

    def reset(self):
        r"""Prints the duration and resets current timer."""
        if self.t_start is None:
            raise RuntimeError("Timer wasn't started.")
        else:
            self.__exit__()
            self.__enter__()


def get_stats_summary(
    stats_list: Union[List[GPUStats], List[CUDAStats]]
) -> Union[GPUStatsSummary, CUDAStatsSummary]:  # pragma: no cover
    r"""Creates a summary of collected runtime and memory statistics.
    Returns a :obj:`GPUStatsSummary` if list of :obj:`GPUStats` was passed,
    otherwise (list of :obj:`CUDAStats` was passed),
    returns a :obj:`CUDAStatsSummary`.

    Args:
        stats_list (Union[List[GPUStats], List[CUDAStats]]): A list of
            :obj:`GPUStats` or :obj:`CUDAStats` objects.
    """
    # calculate common statistics
    kwargs = dict(
        time_mean=float(paddle.to_tensor([s.time for s in stats_list]).mean()),
        time_std=float(paddle.to_tensor([s.time for s in stats_list]).std()),
        max_allocated_gpu=max([s.max_allocated_gpu for s in stats_list]),
        max_reserved_gpu=max([s.max_reserved_gpu for s in stats_list]),
        max_active_gpu=max([s.max_active_gpu for s in stats_list]))

    if all(isinstance(s, CUDAStats) for s in stats_list):
        return CUDAStatsSummary(
            **kwargs,
            min_nvidia_smi_free_cuda=min(
                [s.nvidia_smi_free_cuda for s in stats_list]),
            max_nvidia_smi_used_cuda=max(
                [s.nvidia_smi_used_cuda for s in stats_list]),
        )
    else:
        return GPUStatsSummary(**kwargs)

###############################################################################

def read_from_memlab(line_profiler: Any) -> List[float]:  # pragma: no cover
    from pytorch_memlab.line_profiler.line_records import LineRecords

    # Convert and collect memory statistics
    track_stats = [  # Different statistics can be collected as needed
        'allocated_bytes.all.peak',
        'reserved_bytes.all.peak',
        'active_bytes.all.peak',
    ]

    records = LineRecords(line_profiler._raw_line_records,
                          line_profiler._code_infos)
    stats = records.display(None, track_stats)._line_records
    return [byte_to_megabyte(x) for x in stats.values.max(axis=0).tolist()]


def trace_handler(profiler):
    """Handles the profiling trace and exports it as a JSON file."""
    print_time_total(profiler)
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'timeline' + '.json'
    profiler.export(timeline_file, format="json")


def print_time_total(profiler):
    """Prints the total time summary of profiling events."""
    profiler_summary = sorted_table(profiler.get_summary(), "total", op_detail=True)
    print(profiler_summary)


def rename_profile_file(*args):
    """Renames the exported profiling file with custom arguments for identification."""
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'profile'
    for arg in args:
        timeline_file += '-' + arg
    timeline_file += '.json'
    os.rename('timeline.json', timeline_file)

@contextmanager
def xpu_profile(export_chrome_trace=True):
    with paddle.autograd.profiler_legacy.profile(use_xpu=True) as profile:
        yield
    print(profile.key_averages().table(sort_by='self_xpu_time_total'))
    if export_chrome_trace:
        profile.export_chrome_trace('timeline.json')

@contextmanager
def paddle_profile(export_chrome_trace=True, csv_data=None, write_csv=None):
    """
    A context manager to profile Paddle code execution.

    Args:
        export_chrome_trace (bool): Whether to export the profiling trace as a Chrome-compatible JSON file.
        csv_data (dict): A dictionary to store profiling data for exporting to CSV.
        write_csv (str): If set to 'prof', writes the profiling data to the specified CSV dictionary.
    """
    # Specify profiling targets (CPU, GPU)
    activities = ['cpu']
    if paddle.is_compiled_with_cuda():
        activities.append('gpu')

    profiler = Profiler(
        targets=activities,
        on_trace_ready=trace_handler if export_chrome_trace else print_time_total,
    )

    profiler.start()
    try:
        yield
    finally:
        profiler.stop()

    if csv_data is not None and write_csv == 'prof':
        events = profiler.get_summary(op_detail=True, sort_by='total')
        save_profile_data(csv_data, events, paddle.is_compiled_with_cuda())


def format_prof_time(time):
    """
    Formats profiling time from microseconds to seconds.

    Args:
        time (float): Time in microseconds.

    Returns:
        float: Time in seconds.
    """
    return round(time / 1e6, 3)


def save_profile_data(csv_data, events, use_cuda):
    """
    Saves profiling data to a CSV-compatible dictionary.

    Args:
        csv_data (dict): Dictionary to store profiling data.
        events: Profiling events.
        use_cuda (bool): Whether CUDA events are included in the profiling.
    """
    sum_self_cpu_time_total = sum(
        [event.self_cpu_time for event in events])
    sum_cpu_time_total = sum([event.cpu_time for event in events])
    sum_self_cuda_time_total = sum(
        [event.gpu_time for event in events]) if use_cuda else 0

    for e in events[:5]:  # Save the top 5 most time-consuming operations
        csv_data['NAME'].append(e.op_name)
        csv_data['SELF CPU %'].append(
            round(e.self_cpu_time * 100.0 / sum_self_cpu_time_total, 3))
        csv_data['SELF CPU'].append(format_prof_time(e.self_cpu_time))
        csv_data['CPU TOTAL %'].append(
            round(e.cpu_time * 100.0 / sum_cpu_time_total, 3))
        csv_data['CPU TOTAL'].append(format_prof_time(e.cpu_time))
        csv_data['CPU TIME AVG'].append(format_prof_time(e.cpu_time))
        if use_cuda:
            csv_data['SELF CUDA %'].append(
                e.gpu_time * 100.0 / sum_self_cuda_time_total)
            csv_data['SELF CUDA'].append(format_prof_time(e.gpu_time))
            csv_data['CUDA TOTAL'].append(format_prof_time(e.gpu_time))
            csv_data['CUDA TIME AVG'].append(format_prof_time(e.gpu_time))
        csv_data['# OF CALLS'].append(e.calls)