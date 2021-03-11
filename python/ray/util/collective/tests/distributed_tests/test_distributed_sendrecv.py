"""Test the send/recv API."""
import cupy as cp
import pytest
import ray

from ray.util.collective.tests.util import create_collective_workers


@pytest.mark.parametrize("group_name", ["default", "test", "123?34!"])
@pytest.mark.parametrize("dst_rank", [0, 1, 2, 3])
@pytest.mark.parametrize("src_rank", [0, 1, 2, 3])
@pytest.mark.parametrize("array_size",
                         [2**10, 2**15, 2**20, [2, 2], [5, 9, 10, 85]])
def test_sendrecv(ray_start_distributed_2_nodes_4_gpus, group_name, array_size,
                  src_rank, dst_rank):
    if src_rank == dst_rank:
        return
    world_size = 4
    actors, _ = create_collective_workers(
        num_workers=world_size, group_name=group_name)
    ray.get([
        a.set_buffer.remote(cp.ones(array_size, dtype=cp.float32) * (i + 1))
        for i, a in enumerate(actors)
    ])
    refs = []
    for i in range(world_size):
        refs.append(actors[i].get_buffer.remote())
    refs[src_rank] = actors[src_rank].do_send.remote(group_name, dst_rank)
    refs[dst_rank] = actors[dst_rank].do_recv.remote(group_name, src_rank)
    results = ray.get(refs)
    assert (results[src_rank] == cp.ones(array_size, dtype=cp.float32) *
            (src_rank + 1)).all()
    assert (results[dst_rank] == cp.ones(array_size, dtype=cp.float32) *
            (src_rank + 1)).all()
    ray.get([a.destroy_group.remote(group_name) for a in actors])

@pytest.mark.parametrize("num_calls", [2, 4, 8, 16, 32, 48])
@pytest.mark.parametrize("dst_rank", [0, 1, 2])
@pytest.mark.parametrize("src_rank", [0, 1, 2])
def test_sendrecv_multiple_call(ray_start_distributed_2_nodes_4_gpus, num_calls, dst_rank, src_rank):
    if src_rank == dst_rank:
        return
    world_size = 3
    actors, _ = create_collective_workers(world_size)
    ray.get([
        a.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * (i + 1))
        for i, a in enumerate(actors)
    ])
    for i in range(num_calls):
        refs = []
        for j, actor in enumerate(actors):
            # interleave to create potential synchronization problem
            if i % 2 == 0:
                if j == src_rank:
                    ref = actor.do_send.remote(dst_rank=dst_rank)
                elif j == dst_rank:
                    ref = actor.do_recv.remote(src_rank=src_rank)
                else:
                    continue
                refs.append(ref)
            else:
                if j == dst_rank:
                    ref = actor.do_send.remote(dst_rank=src_rank)
                elif j == src_rank:
                    ref = actor.do_recv.remote(src_rank=dst_rank)
                else:
                    continue
                refs.append(ref)
        results = ray.get(refs)
        ray.get([
            a.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * (j + 1))
            for j, a in enumerate(actors)
        ])
        for j in range(2):
            if i % 2 == 0:
                assert (results[j] == cp.ones((10, ), dtype=cp.float32) *
                    (src_rank + 1)).all()
            else:
                assert (results[j] == cp.ones((10, ), dtype=cp.float32) *
                    (dst_rank + 1)).all()

@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("num_calls", [2, 4, 6, 8, 12])
@pytest.mark.parametrize("dst_rank", [0, 1, 2])
@pytest.mark.parametrize("src_rank", [0, 1, 2])
def test_sendrecv_multiple_group_call(ray_start_distributed_2_nodes_4_gpus, num_groups, num_calls, dst_rank, src_rank):
    if src_rank == dst_rank:
        return
    world_size = 3
    actors, _ = create_collective_workers(world_size)
    ray.wait([
        a.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * (i + 1))
        for i, a in enumerate(actors)
    ])
    for group_name in range(1, num_groups):
        ray.get([
            actor.init_group.remote(world_size, i, group_name=str(group_name))
            for i, actor in enumerate(actors)
        ])
    for _ in range(num_calls):
        for i in range(num_groups):
            refs = []
            group_name = "default" if i == 0 else str(i)
            for j, actor in enumerate(actors):
                # interleave to create potential synchronization problem
                if i % 2 == 0:
                    if j == src_rank:
                        ref = actor.do_send.remote(group_name, dst_rank=dst_rank)
                    elif j == dst_rank:
                        ref = actor.do_recv.remote(group_name, src_rank=src_rank)
                    else:
                        continue
                    refs.append(ref)
                else:
                    if j == dst_rank:
                        ref = actor.do_send.remote(group_name, dst_rank=src_rank)
                    elif j == src_rank:
                        ref = actor.do_recv.remote(group_name, src_rank=dst_rank)
                    else:
                        continue
                    refs.append(ref)
                # If we reset buffers here like:
                #
                # ray.wait([actor.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * (i + 1))])
                # 
                # It will result in deadlock. So instead we reset buffer below.
                # The reason is likely due to ray scheduling internals. The buffer will
                # be used by both the collective call and the buffer reset, which results
                # in the deadlock if we don't synchronize them properly. This deadlock happens
                # with or without multistream, and for both collective and p2p calls.
            results = ray.get(refs)
            ray.get([
                a.set_buffer.remote(cp.ones((10, ), dtype=cp.float32) * (j + 1))
                for j, a in enumerate(actors)
            ])
            for j in range(2):
                if i % 2 == 0:
                    assert (results[j] == cp.ones((10, ), dtype=cp.float32) *
                        (src_rank + 1)).all()
                else:
                    assert (results[j] == cp.ones((10, ), dtype=cp.float32) *
                        (dst_rank + 1)).all()