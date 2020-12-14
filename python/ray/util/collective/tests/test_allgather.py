"""Test the collective allgather API."""
import pytest
import ray

import cupy as cp
import torch

from .util import create_collective_workers


def init_tensors_for_gather_scatter(actors, array_size=10, dtype=cp.float32,
                                    tensor_backend='cupy'):
    world_size = len(actors)
    for i, a in enumerate(actors):
        if tensor_backend == 'cupy':
            t = cp.ones(array_size, dtype=cp.float32) * (i + 1)
        elif tensor_backend == 'torch':
            t = torch.ones(array_size, dtype=torch.float32) * (i + 1)
        else:
            raise RuntimeError("Unsupported tensor backend.")
        ray.wait([a.set_buffer.remote(t)])
    if tensor_backend == 'cupy':
        list_buffer = [cp.ones(array_size, dtype=cp.float32) for _ in range(world_size)]
    elif tensor_backend == 'torch':
        list_buffer = [torch.ones(array_size, dtype=torch.float32) for _ in range(world_size)]
    else:
        raise RuntimeError("Unsupported tensor backend.")
    ray.wait([
        a.set_list_buffer.remote(list_buffer)
        for a in actors
    ])


@pytest.mark.parametrize("tensor_backend", ["cupy", "torch"])
@pytest.mark.parametrize("array_size", [2, 2**5, 2**10, 2**15, 2**20, [2, 2], [5, 5, 5]])
def test_allgather_different_array_size(ray_start_single_node_2_gpus,
                                        array_size,
                                        tensor_backend):
    world_size = 2
    actors, _ = create_collective_workers(world_size)
    init_tensors_for_gather_scatter(actors, array_size=array_size)
    results = ray.get([a.do_allgather.remote() for a in actors])
    for i in range(world_size):
        for j in range(world_size):
            assert (results[i][j] == cp.ones(array_size, dtype=cp.float32) * (j + 1)).all()


@pytest.mark.parametrize("dtype",
                         [cp.uint8, cp.float16, cp.float32, cp.float64])
def test_allreduce_different_dtype(ray_start_single_node_2_gpus, dtype):
    world_size = 2
    actors, _ = create_collective_workers(world_size)
    init_tensors_for_gather_scatter(actors, dtype=dtype)
    results = ray.get([a.do_allgather.remote() for a in actors])
    for i in range(world_size):
        for j in range(world_size):
            assert (results[i][j] == cp.ones(10, dtype=dtype) * (j + 1)).all()


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_unmatched_tensor_list_length(ray_start_single_node_2_gpus, length):
    world_size = 2
    actors, _ = create_collective_workers(world_size)
    list_buffer = [cp.ones(10, dtype=cp.float32) for _ in range(length)]
    ray.wait([
        a.set_list_buffer.remote(list_buffer)
        for a in actors
    ])
    if length != world_size:
        with pytest.raises(RuntimeError):
            ray.get([a.do_allgather.remote() for a in actors])
    else:
        ray.get([a.do_allgather.remote() for a in actors])



@pytest.mark.parametrize("shape", [10, 20, [4, 5], [1, 3, 5, 7]])
def test_unmatched_tensor_shape(ray_start_single_node_2_gpus, shape):
    world_size = 2
    actors, _ = create_collective_workers(world_size)
    init_tensors_for_gather_scatter(actors, array_size=10)
    list_buffer = [cp.ones(shape, dtype=cp.float32) for _ in range(world_size)]
    ray.get([
        a.set_list_buffer.remote(list_buffer)
        for a in actors
    ])
    if shape != 10:
        with pytest.raises(RuntimeError):
            ray.get([a.do_allgather.remote() for a in actors])
    else:
        ray.get([a.do_allgather.remote() for a in actors])

#
# def test_allreduce_torch_cupy(ray_start_single_node_2_gpus):
#     # import torch
#     world_size = 2
#     actors, _ = create_collective_workers(world_size)
#     ray.wait([actors[1].set_buffer.remote(torch.ones(10, ).cuda())])
#     results = ray.get([a.do_allreduce.remote() for a in actors])
#     assert (results[0] == cp.ones((10, )) * world_size).all()
#
#     ray.wait([actors[0].set_buffer.remote(torch.ones(10, ))])
#     ray.wait([actors[1].set_buffer.remote(cp.ones(10, ))])
#     with pytest.raises(RuntimeError):
#         results = ray.get([a.do_allreduce.remote() for a in actors])


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "-x", __file__]))
