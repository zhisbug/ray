import ray
import cupy as cp

import ray.util.collective as collective
from cupy.cuda import Device

@ray.remote(num_gpus=2)
class Worker:
    def __init__(self):
        with Device(0):
            self.send1 = [cp.ones((4, ), dtype=cp.float32), cp.ones((4, ), dtype=cp.float32)] * 2
        with Device(1):
            self.send2 = [cp.ones((4, ), dtype=cp.float32) * 2, cp.ones((4, ), dtype=cp.float32) * 2] * 2
    
        self.recv = []
        with Device(0):
            self.recv.append(cp.zeros((4, ), dtype=cp.float32))

        with Device (1):
            self.recv.append(cp.zeros((4, ), dtype=cp.float32))

    def setup(self, world_size, rank):
        collective.init_collective_group(world_size, rank, "nccl", "179")
        return True

    def compute(self):
        collective.reducescatter_multigpu(self.recv, [self.send1, self.send2], "179")
        return self.recv

    def destroy(self):
        collective.destroy_collective_group("179")


if __name__ == "__main__":

    send = cp.ones((4, ), dtype=cp.float32)

    ray.init(address='auto')

    num_workers = 2
    workers = []
    init_rets = []
    for i in range(num_workers):
        w = Worker.remote()
        workers.append(w)
        init_rets.append(w.setup.remote(num_workers, i))
    a = ray.get(init_rets)
    results = ray.get([w.compute.remote() for w in workers])
    print(results)
    ray.get([w.destroy.remote() for w in workers])
 #   ray.shutdown()
