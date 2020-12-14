import cupy as cp

import ray
import ray.util.collective as col
from ray.util.collective.types import Backend, ReduceOp


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.buffer = cp.ones((10, ), dtype=cp.float32)
        self.list_buffer = [cp.ones((10, ), dtype=cp.float32),
                            cp.ones((10, ), dtype=cp.float32)]

    def init_group(self,
                   world_size,
                   rank,
                   backend=Backend.NCCL,
                   group_name="default"):
        col.init_collective_group(world_size, rank, backend, group_name)
        return True

    def set_buffer(self, data):
        self.buffer = data
        return self.buffer

    def set_list_buffer(self, list_of_arrays):
        self.list_buffer = list_of_arrays
        return self.list_buffer

    def do_allreduce(self, group_name="default", op=ReduceOp.SUM):
        col.allreduce(self.buffer, group_name, op)
        return self.buffer

    def do_reduce(self, group_name="default", dst_rank=0, op=ReduceOp.SUM):
        col.reduce(self.buffer, dst_rank, group_name, op)
        return self.buffer

    def do_broadcast(self, group_name="default", src_rank=0):
        col.broadcast(self.buffer, src_rank, group_name)
        return self.buffer

    def do_allgather(self, group_name="default"):
        col.allgather(self.list_buffer, self.buffer, group_name)
        return self.list_buffer

    def destroy_group(self, group_name="default"):
        col.destroy_collective_group(group_name)
        return True

    def report_rank(self, group_name="default"):
        rank = col.get_rank(group_name)
        return rank

    def report_world_size(self, group_name="default"):
        ws = col.get_world_size(group_name)
        return ws

    def report_nccl_availability(self):
        avail = col.nccl_available()
        return avail

    def report_mpi_availability(self):
        avail = col.mpi_available()
        return avail

    def report_is_group_initialized(self, group_name="default"):
        is_init = col.is_group_initialized(group_name)
        return is_init


def create_collective_workers(num_workers=2, group_name="default", backend="nccl"):
    actors = [Worker.remote() for _ in range(num_workers)]
    world_size = num_workers
    init_results = ray.get([
        actor.init_group.remote(world_size, i, backend, group_name)
        for i, actor in enumerate(actors)
    ])
    return actors, init_results
