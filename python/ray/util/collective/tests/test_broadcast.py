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
@pytest.mark.parametrize("src_rank", [0, 1])
def test_broadcast_different_name(ray_start_single_node_2_gpus, group_name, src_rank):
    world_size = 2
    actors, _ = get_actors_group(num_workers=world_size, group_name=group_name)
    ray.wait([a.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * i) for i, a in enumerate(actors)])
    results = ray.get([a.do_broadcast.remote(group_name, src_rank) for a in actors])
    for i in range(world_size):
        assert (results[i] == cp.ones((10, ), dtype=cp.float32) * src_rank).all()