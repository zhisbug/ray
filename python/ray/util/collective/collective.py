"""APIs exposed under the namespace ray.util.collective."""
import logging

import ray
from ray.util.collective import types
from ray.util.collective.const import NAMED_ACTOR_STORE_SUFFIX

# Get the availability information first by importing information
_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True

try:
    from ray.util.collective.collective_group.mpi_collective_group import MPIGroup
except ImportError:
    _MPI_AVAILABLE = False

try:
    from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
except ImportError:
    _NCCL_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def nccl_available():
    return _NCCL_AVAILABLE


def mpi_available():
    return _MPI_AVAILABLE


class GroupManager(object):
    """
    Use this class to manage the collective groups we created so far;

    """
    def __init__(self):
        """Put some necessary meta information here."""
        self._name_group_map = {}
        self._group_name_map = {}

    def create_collective_group(self,
                                backend,
                                world_size,
                                rank,
                                group_name):
        """
        The entry to create new collective groups and register in the manager.

        Put the registration and the group information into the manager metadata as well.
        """
        backend = types.Backend(backend)
        if backend == types.Backend.MPI:
            raise NotImplementedError()
        elif backend == types.Backend.NCCL:
            # create the ncclUniqueID
            if rank == 0:
                import cupy.cuda.nccl as nccl
                group_uid = nccl.get_unique_id()
                store_name = group_name + NAMED_ACTOR_STORE_SUFFIX

                # Avoid a potential circular dependency in ray/actor.py
                from ray.util.collective.util import NCCLUniqueIDStore
                store = NCCLUniqueIDStore.options(name=store_name, lifetime="detached").remote(store_name)
                ray.wait([store.set_id.remote(group_uid)])

            logger.debug('creating NCCL group: {}'.format(group_name))
            g = NCCLGroup(world_size, rank, group_name)
            self._name_group_map[group_name] = g
            self._group_name_map[g] = group_name
        return self._name_group_map[group_name]

    def is_group_exist(self, group_name):
        if group_name in self._name_group_map:
            return True
        return False

    def get_group_by_name(self, group_name):
        """Get the collective group handle by its name."""
        if group_name not in self._name_group_map:
            return None
        return self._name_group_map[group_name]

    def destroy_collective_group(self, group_name):
        """Group destructor."""
        if group_name not in self._name_group_map:
            logger.warning('The group {} does not exist'.format(group_name))
            return

        # release the collective group resource
        g = self._name_group_map[group_name]

        rank = g.rank
        backend = g.backend()

        # clean up the dicts
        del self._group_name_map[g]
        del self._name_group_map[group_name]

        if backend == types.Backend.NCCL:
            # release the named actor
            if rank == 0:
                store_name = group_name + NAMED_ACTOR_STORE_SUFFIX
                store = ray.get_actor(store_name)
                ray.wait([store.__ray_terminate__.remote()])
                ray.kill(store)
        g.destroy()


_group_mgr = GroupManager()


def init_collective_group(backend,
                          world_size,
                          rank,
                          group_name='default'):
    """
    Initialize a collective group inside an actor process.

    This is an
    Args:
        backend:
        world_size:
        rank:
        group_name:

    Returns:
        None
    """
    backend = types.Backend(backend)
    _check_backend_availability(backend)
    global _group_mgr
    # TODO(Hao): implement a group auto-counter.
    if not group_name:
        raise ValueError('group_name: {},  needs to be a string.'.format(group_name))

    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError('Trying to initialize a group twice.')

    assert(world_size > 0)
    assert(rank >= 0 )
    assert(rank < world_size)
    _group_mgr.create_collective_group(backend, world_size, rank, group_name)


def allreduce(tensor,
              group_name,
              op=types.ReduceOp.SUM):
    """
    Collective allreduce the tensor across the group with name group_name.

    Args:
        tensor:
        group_name (string):
        op:

    Returns:
        None
    """
    g = _check_and_get_group(group_name)
    opts = types.AllReduceOptions
    opts.reduceOp = op
    g.allreduce(tensor, opts)


def barrier(group_name):
    """
    Barrier all collective process in the group with name group_name.

    Args:
        group_name (string):

    Returns:
        None
    """
    g = _check_and_get_group(group_name)
    g.barrier()


def _check_and_get_group(group_name):
    """Check the existence and return the group handle."""
    global _group_mgr
    if not _group_mgr.is_group_exist(group_name):
        raise ValueError('The collective group {} is not initialized.'.format(group_name))
    # TODO(Hao): check if this rank is in the group.
    g = _group_mgr.get_group_by_name(group_name)
    return g


def _check_backend_availability(backend: types.Backend):
    """Check whether the backend is available."""
    if backend == types.Backend.MPI:
        if not mpi_available():
            raise RuntimeError('MPI is not available.')
    elif backend == types.Backend.NCCL:
        if not nccl_available():
            raise RuntimeError('NCCL is not available.')
