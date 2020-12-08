"""Test the reduce API."""
import pytest
import cupy as cp
import ray

from .util import Worker


def get_actors_group(num_workers=2, group_name="default", backend="nccl"):
    actors = [Worker.remote() for i in range(num_workers)]
    world_size = num_workers
    init_results = ray.get([
        actor.init_group.remote(world_size, i, backend, group_name)
        for i, actor in enumerate(actors)
    ])
    return actors, init_results


@pytest.mark.parametrize("group_name", ["default", "test", "123?34!"])
@pytest.mark.parametrize("dst_rank", [0, 1])
def test_reduce_different_name(ray_start_single_node_2_gpus, group_name, dst_rank):
    world_size = 2
    actors, _ = get_actors_group(num_workers=world_size, group_name=group_name)
    results = ray.get([a.do_reduce.remote(group_name, dst_rank) for a in actors])
    for i in range(world_size):
        if i == dst_rank:
            assert (results[i] == cp.ones((10,), dtype=cp.float32) * world_size).all()
        else:
            assert (results[i] == cp.ones((10, ), dtype=cp.float32)).all()

# @pytest.mark.parametrize("array_size", [2, 2**5, 2**10, 2**15, 2**20])
# def test_allreduce_different_array_size(ray_start_single_node_2_gpus,
#                                         array_size):
#     world_size = 2
#     actors, _ = get_actors_group(world_size)
#     ray.wait([
#         a.set_buffer.remote(cp.ones(array_size, dtype=cp.float32))
#         for a in actors
#     ])
#     results = ray.get([a.do_work.remote() for a in actors])
#     assert (results[0] == cp.ones(
#         (array_size, ), dtype=cp.float32) * world_size).all()
#     assert (results[1] == cp.ones(
#         (array_size, ), dtype=cp.float32) * world_size).all()
#
#
# def test_allreduce_destroy(ray_start_single_node_2_gpus,
#                            backend="nccl",
#                            group_name="default"):
#     world_size = 2
#     actors, _ = get_actors_group(world_size)
#
#     results = ray.get([a.do_work.remote() for a in actors])
#     assert (results[0] == cp.ones((10, ), dtype=cp.float32) * world_size).all()
#     assert (results[1] == cp.ones((10, ), dtype=cp.float32) * world_size).all()
#
#     # destroy the group and try do work, should fail
#     ray.wait([a.destroy_group.remote() for a in actors])
#     with pytest.raises(RuntimeError):
#         results = ray.get([a.do_work.remote() for a in actors])
#
#     # reinit the same group and all reduce
#     ray.get([
#         actor.init_group.remote(world_size, i, backend, group_name)
#         for i, actor in enumerate(actors)
#     ])
#     results = ray.get([a.do_work.remote() for a in actors])
#     assert (results[0] == cp.ones(
#         (10, ), dtype=cp.float32) * world_size * 2).all()
#     assert (results[1] == cp.ones(
#         (10, ), dtype=cp.float32) * world_size * 2).all()
#
#
# def test_allreduce_multiple_group(ray_start_single_node_2_gpus,
#                                   backend="nccl",
#                                   num_groups=5):
#     world_size = 2
#     actors, _ = get_actors_group(world_size)
#     for group_name in range(1, num_groups):
#         ray.get([
#             actor.init_group.remote(world_size, i, backend, str(group_name))
#             for i, actor in enumerate(actors)
#         ])
#     for i in range(num_groups):
#         group_name = "default" if i == 0 else str(i)
#         results = ray.get([a.do_work.remote(group_name) for a in actors])
#         assert (results[0] == cp.ones(
#             (10, ), dtype=cp.float32) * (world_size**(i + 1))).all()
#
#
# def test_allreduce_different_op(ray_start_single_node_2_gpus):
#     world_size = 2
#     actors, _ = get_actors_group(world_size)
#
#     # check product
#     ray.wait([
#         a.set_buffer.remote(cp.ones(10, dtype=cp.float32) * (i + 2))
#         for i, a in enumerate(actors)
#     ])
#     results = ray.get([a.do_work.remote(op=ReduceOp.PRODUCT) for a in actors])
#     assert (results[0] == cp.ones((10, ), dtype=cp.float32) * 6).all()
#     assert (results[1] == cp.ones((10, ), dtype=cp.float32) * 6).all()
#
#     # check min
#     ray.wait([
#         a.set_buffer.remote(cp.ones(10, dtype=cp.float32) * (i + 2))
#         for i, a in enumerate(actors)
#     ])
#     results = ray.get([a.do_work.remote(op=ReduceOp.MIN) for a in actors])
#     assert (results[0] == cp.ones((10, ), dtype=cp.float32) * 2).all()
#     assert (results[1] == cp.ones((10, ), dtype=cp.float32) * 2).all()
#
#     # check max
#     ray.wait([
#         a.set_buffer.remote(cp.ones(10, dtype=cp.float32) * (i + 2))
#         for i, a in enumerate(actors)
#     ])
#     results = ray.get([a.do_work.remote(op=ReduceOp.MAX) for a in actors])
#     assert (results[0] == cp.ones((10, ), dtype=cp.float32) * 3).all()
#     assert (results[1] == cp.ones((10, ), dtype=cp.float32) * 3).all()
#
#
# @pytest.mark.parametrize("dtype",
#                          [cp.uint8, cp.float16, cp.float32, cp.float64])
# def test_allreduce_different_dtype(ray_start_single_node_2_gpus, dtype):
#     world_size = 2
#     actors, _ = get_actors_group(world_size)
#     ray.wait([a.set_buffer.remote(cp.ones(10, dtype=dtype)) for a in actors])
#     results = ray.get([a.do_work.remote() for a in actors])
#     assert (results[0] == cp.ones((10, ), dtype=dtype) * world_size).all()
#     assert (results[1] == cp.ones((10, ), dtype=dtype) * world_size).all()
#
#
# def test_allreduce_torch_cupy(ray_start_single_node_2_gpus):
#     # import torch
#     world_size = 2
#     actors, _ = get_actors_group(world_size)
#     ray.wait([actors[1].set_buffer.remote(torch.ones(10, ).cuda())])
#     results = ray.get([a.do_work.remote() for a in actors])
#     assert (results[0] == cp.ones((10, )) * world_size).all()
#
#     ray.wait([actors[0].set_buffer.remote(torch.ones(10, ))])
#     ray.wait([actors[1].set_buffer.remote(cp.ones(10, ))])
#     with pytest.raises(RuntimeError):
#         results = ray.get([a.do_work.remote() for a in actors])
