import ray
tensor0 = None
tensor1 = None
tensor2 = None
tensor3 = None
options = None


@ray.remote(num_gpus=4)
class Worker:
    def __init__(self):
        pass

actors = [Worker.remote() for i in range(16)]
# This one is faster than one tensor per process due to NCCL optimization.
for actor in actors:
    actor.allreduce_multi_gpu.remote(
        [tensor0, tensor1, tensor2, tensor3], options, ...)



