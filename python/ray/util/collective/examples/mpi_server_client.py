"""
spawn with ray
"""


import mpi4py.MPI as MPI
import ray
import time
import datetime
import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

NAMED_ACTOR_STORE_SUFFIX = "_unique_id_actor"

def get_mpi_store_name(group_name):
    """Generate the unique name for the NCCLUniqueID store (named actor)."""
    if not group_name:
        raise ValueError("group_name is None.")
    return group_name + NAMED_ACTOR_STORE_SUFFIX

class Rendezvous:
    def __init__(self, group_name):
        if not group_name:
            raise ValueError('Empty meeting point.')
        self._group_name = group_name
        self._store_name = None
        self._store = None

    def meet(self, timeout=180):
        """Meet at the named actor store."""
        if timeout is not None and timeout < 0:
            raise ValueError("The 'timeout' argument must be nonnegative. "
                             f"Received {timeout}")
        self._store_name = get_mpi_store_name(self._group_name)
        timeout_delta = datetime.timedelta(seconds=timeout)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            try:
                logger.debug("Trying to meet at the store '{}'".format(self._store_name))
                self._store = ray.get_actor(self._store_name)
            except ValueError:
                logger.debug("Failed to meet at the store '{}'."
                              "Trying again...".format(self._store_name))
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            logger.debug("Successful rendezvous!")
            break
        if not self._store:
            raise RuntimeError("Unable to meet other processes "
                               "at the rendezvous store.")

    @property
    def store(self):
        return self._store


    def get_mpi_id(self, timeout=180):
        """Get the MPIUniqueID from the store."""
        if not self._store:
            raise ValueError("Rendezvous store is not setup.")
        uid = None
        timeout_delta = datetime.timedelta(seconds=timeout)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            uid = ray.get(self._store.get_id.remote())
            if not uid:
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            break
        if not uid:
            raise RuntimeError(
                "Unable to get the MPIUniqueID from the store.")
        return uid


@ray.remote
class MPIUniqueIDStore(object):
    """MPIUniqueID Store as a named actor."""

    def __init__(self, name):
        self.name = name
        self.mpi_id = None

    def set_id(self, uid):
        self.mpi_id = uid
        return self.mpi_id

    def get_id(self):
        if not self.mpi_id:
            print(
                "The MPI ID has not been set yet for store {}".format(
                    self.name))
        return self.mpi_id

@ray.remote(num_cpus=1)
def get_ranks(world_size, rank):
    mpi_comm = MPI.COMM_WORLD
    group_name = 'mpi'
    # server = get_mpi_store_name(group_name)
    server = "123456"
    if rank == 0:
        port = MPI.Open_port()

        print('creating info')
        info = MPI.Info.Create()
        info.Set("ompi_global_scope", "true")
        info.Set("ompi-server", "true")

        print('publishing a name')
        MPI.Publish_name(server, port, info)

        # print('set a store')
        # group_uid = port
        # store_name = get_mpi_store_name(group_name)
        # store = MPIUniqueIDStore.options(name=store_name, lifetime="detached").remote(store_name)
        # print('remote seting port/uid:', port)
        # ray.wait([store.set_id.remote(group_uid)])

        print('accepting:', port)
        inter_comm = mpi_comm.Accept(port)
        # print('mergeing')
        inter_comm.Barrier()
        merge_comm = inter_comm.Merge()
        print("rank {}, merge_world_size: {}, merge_true_rank: {} ".format(rank, merge_comm.Get_size(), merge_comm.Get_rank()))
    else:
        time.sleep(1)
        # _rendezvous = Rendezvous(group_name)
        # print('meeting')
        # _rendezvous.meet()
        # _mpi_port = _rendezvous.get_mpi_id()

        _mpi_port = MPI.Lookup_name(server)
        # _mpi_port.Barrier()
        print('connecting:', _mpi_port)
        # connect to the server
        inter_comm = mpi_comm.Connect(_mpi_port)
        # connect_comm = mpi_comm.Connect(_mpi_port)
    print("rank {}, world_size: {}, true_rank: {} ".format(rank, inter_comm.Get_size(), inter_comm.Get_rank()))


if __name__ == "__main__":
    world_size = 2
    ray.init(num_cpus=world_size)
    include_dashboard=False
    ray.get([get_ranks.remote(world_size, i) for i in range(world_size)])
