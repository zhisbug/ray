"""APIs exposed under the namespace ray.util.collective."""
import logging
from collections import defaultdict
import ray
from ray.util.collective import types
from ray.util.collective.const import NAMED_ACTOR_STORE_SUFFIX
import ray.worker
import cupy.cuda.nccl as nccl

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

logging.getLogger().setLevel(logging.DEBUG)

def nccl_available():
    """Check whether nccl is available."""
    return _NCCL_AVAILABLE


def mpi_available():
    """Check whether mpi is available."""
    return _MPI_AVAILABLE


def _backend_check(backend):
    if backend == 'mpi':
        if not mpi_available():
            raise RuntimeError()
        raise NotImplementedError()
    elif backend == 'nccl':
        if not nccl_available():
            raise RuntimeError()

@ray.remote
class NCCLUniqueIDStore():
    """NCCLUniqueID. This class should be used as a named actor."""
    def __init__(self, name):
        self.name = name
        self.nccl_id = None

    def set_id(self, uid):
        """Set nccl id of the store."""
        self.nccl_id = uid
        return self.nccl_id

    def get_id(self):
        """Get nccl id from the store."""
        if not self.nccl_id:
            logging.warning('The NCCL ID has not been set yet for store {}'.format(self.name))
        return self.nccl_id

class GroupManager():
    """
    Use this class to manage the collective groups we created so far;

    """
    def __init__(self):
        """Put some necessary meta information here."""
        self._name_group_map = {}
        self._group_name_map = {}
        self._group_actor_map = defaultdict(defaultdict)

    def create_collective_group(self,
                                backend,
                                world_size,
                                rank,
                                group_name):
        """
        The only entry to create new collective groups and register to the manager.

        Put the registration and the group information into the manager metadata as well.
        """
        if backend == 'mpi':
            raise NotImplementedError()
        elif backend == 'nccl':
            # create the ncclUniqueID
            if rank == 0:
                group_uid = nccl.get_unique_id()
                store_name = group_name + NAMED_ACTOR_STORE_SUFFIX

                store = NCCLUniqueIDStore.options(name=store_name,
                                                  lifetime="detached").remote(store_name)

                ray.wait([store.set_id.remote(group_uid)])

            logging.debug('creating NCCL group: {}'.format(group_name))
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
            logging.warning('The group {} does not exist'.format(group_name))
            return

        # release the collective group resource
        g = self._name_group_map[group_name]

        rank = g.rank
        backend = g.backend()

        # clean up the dicts
        del self._group_name_map[g]
        del self._name_group_map[group_name]

        if backend == 'nccl':
            # release the named actor
            if rank == 0:
                store_name = group_name + NAMED_ACTOR_STORE_SUFFIX
                store = ray.get_actor(store_name)
                ray.wait([store.__ray_terminate__.remote()])
                ray.kill(store)
        g.destroy()

global _group_mgr
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

    """
    _backend_check(backend)
    # TODO(Hao): implement a group auto-counter.
    if not group_name:
        raise ValueError('group_name: {},  needs to be a string.'.format(group_name))

    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError('Trying to initialize a group twice.')
    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    _group_mgr.create_collective_group(backend, world_size, rank, group_name)

@ray.remote
class Info:
    """Store the collective information for groups created through declare_collective_group().
       Should be used as a NamedActor."""

    def __init__(self):
        self.ids = None
        self.world_size = -1
        self.rank = -1
        self.backend = None

    def set_info(self, ids, world_size, rank, backend):
        """Store collective information."""
        self.ids = ids
        self.world_size = world_size
        self.rank = rank
        self.backend = backend

    def get_info(self):
        """Get previously stored collective information."""
        return self.ids, self.world_size, self.rank, self.backend

def declare_collective_group(actors, group_options):
    """
    Declare a list of actors in a collective group with group options. This function
    should be called in a driver process.
    Args:
        actors (list): a list of actors to be set in a collective group.
        group_options (dict): a dictionary that contains group_name(str), world_size(int),
                              rank(list of int, e.g. [0,1] means the first actor is rank 0, and
                              the second actor is rank 1), backend(str)

    Returns:

    """
    try:
        group_name = group_options["group_name"]
        world_size = group_options["world_size"]
        rank = group_options["rank"]
        backend = group_options["backend"]
    except:
        raise ValueError("group options incomplete.")

    _backend_check(backend)
    name = "info" + group_name
    try:
        ray.get_actor(name)
        raise RuntimeError('Trying to initialize a group twice.')
    except:
        pass

    if len(rank) != len(actors):
        raise RuntimeError("Each actor should correspond to one rank.")

    if set(rank) != set(range(len(rank))):
        raise RuntimeError("Rank must be a permutation from 0 to len-1.")

    assert world_size > 0
    assert all(rank) >= 0 and all(rank) < world_size

    # store the information into a NamedActor that can be accessed later/
    name = "info" + group_name
    actors_id = [a._ray_actor_id for a in actors]
    info = Info.options(name=name, lifetime="detached").remote()
    ray.wait([info.set_info.remote(actors_id, world_size, rank, backend)])

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
    #global _group_mgr
    if not _group_mgr.is_group_exist(group_name):
        # try loading from remote info store
        try:
            # if the information is stored in an Info object, get and create the group.
            name = "info" + group_name
            mgr = ray.get_actor(name=name)
            ids, world_size, rank, backend = ray.get(mgr.get_info.remote())
            worker = ray.worker.global_worker
            id_ = worker.core_worker.get_actor_id()
            r = rank[ids.index(id_)]
            _group_mgr.create_collective_group(backend, world_size, r, group_name)
        except:
            raise ValueError('The collective group {} is not initialized.'.format(group_name))
    # TODO(Hao): check if this rank is in the group.
    g = _group_mgr.get_group_by_name(group_name)
    return g
