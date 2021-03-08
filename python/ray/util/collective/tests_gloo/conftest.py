"""Some fixtures for collective tests."""
import logging

import pytest
import ray

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


@pytest.fixture
def ray_start_single_node():
    address_info = ray.init()
    yield address_info
    ray.shutdown()


# Hao: this fixture is a bit tricky.
# I use a bash script to start a ray cluster on
# my own on-premise cluster before run this fixture.
@pytest.fixture
def ray_start_distributed_2_nodes():
    # The cluster has a setup of 2 nodes.

    ray.init("auto")
    yield
    ray.shutdown()
