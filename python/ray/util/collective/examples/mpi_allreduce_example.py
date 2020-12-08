
import ray
import numpy as np
import torch

import ray.util.collective as collective
@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, world_size):
        self.send = [torch.ones((4,), dtype=torch.float32)] * world_size

    def setup(self, world_size, rank):
        collective.init_collective_group(world_size, rank, 'mpi', 'default')
        print()
        return True

    def compute(self, i):
        collective.allreduce(self.send[i], 'default')
        return self.send[i]

    def destroy(self):
        collective.destroy_group('')

if __name__ == "__main__":
    ray.init(num_gpus=3, num_cpus=10)

    num_workers = 3
    workers = []
    init_rets = []
    for i in range(num_workers):
        w = Worker.remote(num_workers)
        workers.append(w)
        init_rets.append(w.setup.remote(num_workers, i))
    m = ray.get(init_rets)
    results = ray.get([w.compute.remote(i) for i,w in enumerate(workers)])
    print(results)
    ray.shutdown()