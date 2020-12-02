"""APIs exposed under the namespace ray.util.collective."""
import logging

import ray
from ray.util.collective import types
from ray.util.collective.const import NAMED_ACTOR_STORE_SUFFIX
from collections import defaultdict
import ray.worker

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
    return _NCCL_AVAILABLE


def mpi_available():
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
class NCCLUniqueIDStore(object):
    """NCCLUniqueID. This class should be used as a named actor."""
    def __init__(self, name):
        self.name = name
        self.nccl_id = None

    def set_id(self, uid):
        self.nccl_id = uid
        return self.nccl_id

    def get_id(self):
        if not self.nccl_id:
            logging.warning('The NCCL ID has not been set yet for store {}'.format(self.name))
        return self.nccl_id

class GroupManager(object):
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
                                group_name,
                                actors=None):
        """
        The only entry to create new collective groups and register to the manager.

        Put the registration and the group information into the manager metadata as well.
        """
        if backend == 'mpi':
            raise NotImplementedError()
        elif backend == 'nccl':
            # create the ncclUniqueID
            import cupy.cuda.nccl as nccl
            if actors is None:
                if rank == 0:
                    import cupy.cuda.nccl as nccl
                    group_uid = nccl.get_unique_id()
                    store_name = group_name + NAMED_ACTOR_STORE_SUFFIX

                    store = NCCLUniqueIDStore.options(name=store_name, lifetime="detached").remote(store_name)
                    ray.wait([store.set_id.remote(group_uid)])

                logging.debug('creating NCCL group: {}'.format(group_name))
                g = NCCLGroup(world_size, rank, group_name)
                self._name_group_map[group_name] = g
                self._group_name_map[g] = group_name
            # if we know all the actors
            else:
                raise NotImplementedError()
                group_uid = nccl.get_unique_id()
                self._name_group_map[group_name] = []
                for i in range(len(rank)):
                    logging.debug('creating NCCL group: {}, rank: {}'.format(group_name, rank[i]))
                    g = NCCLGroup(world_size, rank[i], group_name, actors=True)
                    self._group_actor_map[group_name][actors[i]._ray_actor_id] = g
                    logging.debug(f"actor id is {actors[i]._ray_actor_id}")
                    self._name_group_map[group_name].append(g)
                    print(self._name_group_map)
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

@ray.remote(num_gpus=0.1)
class GroupManager_2(object):
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
                                group_name,
                                actors):
        """
        The only entry to create new collective groups and register to the manager.

        Put the registration and the group information into the manager metadata as well.
        """
        if backend == 'mpi':
            raise NotImplementedError()
        elif backend == 'nccl':
            # create the ncclUniqueID
            import cupy.cuda.nccl as nccl
            group_uid = nccl.get_unique_id()
            self._name_group_map[group_name] = []
            for i in range(len(rank)):
                logging.debug('creating NCCL group: {}, rank: {}'.format(group_name, rank[i]))
                g = NCCLGroup(world_size, rank[i], group_name, uid=group_uid)
                self._group_actor_map[group_name][actors[i]._ray_actor_id] = g
                logging.debug(f"actor id is {actors[i]._ray_actor_id}")
                self._name_group_map[group_name].append(g)
                print(self._name_group_map)
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

    def get_group_by_id(self, group_name, id_):
        """Get the collective group handle by its name and id."""
        return self._group_actor_map[group_name][id_]

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
                store_name = group_name + types.named_actor_suffix
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
    assert(world_size > 0)
    assert(rank >= 0 )
    assert(rank < world_size)
    _group_mgr.create_collective_group(backend, world_size, rank, group_name)

global _group_mgr_2

def declare_collective_group(actors, group_options):
    """
    # Frontend API #2:
    # This API is supported to work in the driver program - the users declare a list of actors as a collective group
    # @Dacheng: This API is not in the right shape, need to work with ray.remote(), please figure out.
    Args:
        actors:
        group_options:

    Returns:

    """
    try:
        _group_mgr_2 = GroupManager_2.options(name="GM", lifetime="detached").remote()
    except:
        _group_mgr_2 = ray.get_actor(name="GM")
    try:
        group_name = group_options["group_name"]
        world_size = group_options["world_size"]
        rank = group_options["rank"]
        backend = group_options["backend"]
    except:
        raise ValueError("group options incomplete.")
    
    _backend_check(backend)
    if ray.get(_group_mgr_2.is_group_exist.remote(group_name)):
        raise RuntimeError('Trying to initialize a group twice.')
 
    if len(rank) != len(actors):
        raise RuntimeError("Each actor should correspond to one rank.")
    
    if set(rank) != set(range(len(rank))):
        raise RuntimeError("Rank must be a permutation from 0 to len-1.")
    
    assert(world_size > 0)
    assert(all(rank) >= 0 and all(rank) < world_size)

    res = _group_mgr_2.create_collective_group.remote(backend, world_size, rank, group_name, actors=actors)
    ray.get(res)

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
    # if this group is created through declare()
    if not local_GM:
        g.allreduce(tensor, opts)
    else:
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
    global local_GM
    local_GM = True
    if not _group_mgr.is_group_exist(group_name):
        local_GM = False
        global _group_mgr_2
        _group_mgr_2 = ray.get_actor("GM")
        if not ray.get(_group_mgr_2.is_group_exist.remote(group_name)):
            raise ValueError('The collective group {} is not initialized.'.format(group_name))
    # TODO(Hao): check if this rank is in the group.
    if local_GM:
        g = _group_mgr.get_group_by_name(group_name)
    else:
        worker = ray.worker.global_worker
        id_ = worker.core_worker.get_actor_id()
        g = ray.get(_group_mgr_2.get_group_by_id.remote(group_name, id_))
        print(g)
    return g
