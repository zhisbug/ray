"""Some fixtures for collective tests."""
import pytest

import ray
from ray.util.collective.const import get_nccl_store_name
from ray.util.collective.collective_group.nccl_collective_group \
    import _get_comm_key_from_devices, _get_comm_key_send_recv


# TODO (Hao): remove this clean_up function as it sometimes crashes Ray.
def clean_up():
    group_names = ["default", "test", "123?34!", "default2", "random"]
    group_names.extend([str(i) for i in range(10)])
    max_world_size = 4

    all_keys = []
    for name in group_names:
        devices = [0]
        collective_communicator_key = _get_comm_key_from_devices(devices)
        all_keys.append(collective_communicator_key + "@" + name)
        for i in range(max_world_size):
            for j in range(max_world_size):
                if i < j:
                    p2p_communicator_key = _get_comm_key_send_recv(i, 0, j, 0)
                    all_keys.append(p2p_communicator_key + "@" + name)
    for group_key in all_keys:
        store_name = get_nccl_store_name(group_key)
        try:
            actor = ray.get_actor(store_name)
        except ValueError:
            actor = None
        if actor:
            ray.kill(actor)


@pytest.fixture
def ray_start_single_node_2_gpus():
    # Please start this fixture in a cluster with 2 GPUs.
    address_info = ray.init(num_gpus=2)
    yield address_info
    ray.shutdown()


# Hao: this fixture is a bit tricky.
# I use a bash script to start a ray cluster on
# my own on-premise cluster before run this fixture.
@pytest.fixture
def ray_start_distributed_2_nodes_4_gpus():
    ray.init("auto")
    yield
    clean_up()
    ray.shutdown()
