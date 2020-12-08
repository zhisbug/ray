import cupy as cp

import ray
import ray.util.collective as col
from ray.util.collective.types import Backend, ReduceOp


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.buffer = cp.ones((10, ), dtype=cp.float32)

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

    def do_work(self, group_name="default", op=ReduceOp.SUM):
        col.allreduce(self.buffer, group_name, op)
        return self.buffer

    def do_reduce(self, group_name="default", dst_rank=0, op=ReduceOp.SUM):
        col.reduce(self.buffer, group_name, dst_rank, op)
        return self.buffer

    def do_broadcast(self, group_name="default", src_rank=0):
        col.broadcast(self.buffer, group_name, src_rank)
        return self.buffer

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


def get_actors_group(num_workers=2, group_name="default", backend="nccl"):
    actors = [Worker.remote() for i in range(num_workers)]
    world_size = num_workers
    init_results = ray.get([
        actor.init_group.remote(world_size, i, backend, group_name)
        for i, actor in enumerate(actors)
    ])
    return actors, init_results

ray.init(num_gpus=2)
world_size = 2
print('reach heree..0')
src_rank = 0
group_name = 'default'
actors, _ = get_actors_group(num_workers=world_size, group_name=group_name)
ray.wait([a.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * i) for i, a in enumerate(actors)])
results = ray.get([a.do_broadcast.remote(group_name, src_rank) for a in actors])
for i in range(world_size):
    assert (results[i] == cp.ones((10, ), dtype=cp.float32) * src_rank).all()