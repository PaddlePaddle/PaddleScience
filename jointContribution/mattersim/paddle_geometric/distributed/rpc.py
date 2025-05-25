import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import paddle.distributed as dist
from paddle_geometric.distributed.dist_context import DistContext, DistRole

try:
    from paddle.distributed import rpc_is_initialized
except ImportError:
    def rpc_is_initialized() -> bool:
        return False

_rpc_init_lock = threading.RLock()

def rpc_require_initialized(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if not rpc_is_initialized():
            raise RuntimeError("RPC is not initialized.")
        return func(*args, **kwargs)
    return wrapper

@rpc_require_initialized
def global_all_gather(obj: Any) -> List[Any]:
    """Gathers objects from all groups in a list."""
    return dist.rpc.all_gather(obj)

@rpc_require_initialized
def global_barrier():
    """Barrier function for all RPC processes."""
    try:
        global_all_gather(obj=None)
    except RuntimeError:
        logging.error('Failed to respond to global barrier')

def init_rpc(
    current_ctx: DistContext,
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 16,
    rpc_timeout: float = 240.0,
    rpc_worker_names: Optional[Dict[DistRole, List[str]]] = None,
):
    with _rpc_init_lock:
        if rpc_is_initialized():
            return

        if current_ctx is None:
            raise RuntimeError("'dist_context' has not been set in 'init_rpc'")

        options = {
            "transports": ['tcp'],
            "num_threads": num_rpc_threads,
            "timeout": rpc_timeout,
            "init_method": f"tcp://{master_addr}:{master_port}"
        }
        dist.rpc.init_rpc(
            name=current_ctx.worker_name,
            rank=current_ctx.global_rank,
            world_size=current_ctx.global_world_size,
            options=options
        )

        global_barrier()

def shutdown_rpc(id: str = None, graceful: bool = True):
    with _rpc_init_lock:
        if rpc_is_initialized():
            logging.info(f"Shutting down RPC in {id} gracefully.")
            dist.rpc.shutdown(graceful)
        else:
            logging.info(f'RPC in {id} not initialized.')

class RPCRouter:
    """Router to retrieve workers based on partition ID."""
    def __init__(self, partition_to_workers: List[List[str]]):
        for pid, rpc_worker_list in enumerate(partition_to_workers):
            if len(rpc_worker_list) == 0:
                raise ValueError('No RPC worker in worker list')
        self.partition_to_workers = partition_to_workers
        self.rpc_worker_indices = [0 for _ in range(len(partition_to_workers))]

    def get_to_worker(self, partition_idx: int) -> str:
        rpc_worker_list = self.partition_to_workers[partition_idx]
        worker_idx = self.rpc_worker_indices[partition_idx]
        router_worker = rpc_worker_list[worker_idx]
        self.rpc_worker_indices[partition_idx] = (worker_idx + 1) % len(rpc_worker_list)
        return router_worker

@rpc_require_initialized
def rpc_partition_to_workers(
    current_ctx: DistContext,
    num_partitions: int,
    current_partition_idx: int,
) -> List[List[str]]:
    """Maps partitions to workers through `all_gather`."""
    partition_to_workers = [[] for _ in range(num_partitions)]
    gathered_results = global_all_gather((current_ctx.role, num_partitions, current_partition_idx))
    for worker_name, (role, nparts, idx) in gathered_results.items():
        partition_to_workers[idx].append(worker_name)
    return partition_to_workers

class RPCCallBase(ABC):
    """Base class for RPC call wrappers."""
    @abstractmethod
    def rpc_sync(self, *args, **kwargs):
        pass

    @abstractmethod
    def rpc_async(self, *args, **kwargs):
        pass

_rpc_call_lock = threading.RLock()
_rpc_call_id: int = 0
_rpc_call_pool: Dict[int, RPCCallBase] = {}

@rpc_require_initialized
def rpc_register(call: RPCCallBase) -> int:
    """Registers an RPC call."""
    global _rpc_call_id, _rpc_call_pool

    with _rpc_call_lock:
        call_id = _rpc_call_id
        _rpc_call_id += 1
        if call_id in _rpc_call_pool:
            raise RuntimeError("Registered function twice in 'rpc_register'")
        _rpc_call_pool[call_id] = call

    return call_id

def _rpc_async_call(call_id: int, *args, **kwargs):
    """Entry point for asynchronous RPC calls."""
    return _rpc_call_pool.get(call_id).rpc_async(*args, **kwargs)

@rpc_require_initialized
def rpc_async(worker_name: str, call_id: int, args=None, kwargs=None):
    """Performs an asynchronous RPC request."""
    return dist.rpc.rpc_async(
        worker_name,
        _rpc_async_call,
        args=(call_id, *args),
        kwargs=kwargs,
    )

def _rpc_sync_call(call_id: int, *args, **kwargs):
    """Entry point for synchronous RPC calls."""
    return _rpc_call_pool.get(call_id).rpc_sync(*args, **kwargs)

@rpc_require_initialized
def rpc_sync(worker_name: str, call_id: int, args=None, kwargs=None):
    """Performs a synchronous RPC request."""
    future = dist.rpc.rpc_async(
        worker_name,
        _rpc_sync_call,
        args=(call_id, *args),
        kwargs=kwargs,
    )
    return future.wait()
