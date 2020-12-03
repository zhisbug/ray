import ray
import os
ray.init()

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self):
        pass

    def compute(self):
        return os.environ["1"]

# might work?
worker = Actor.options(override_environment_variables={"1" : "2"}).remote()
print(ray.get(worker.compute.remote()))
