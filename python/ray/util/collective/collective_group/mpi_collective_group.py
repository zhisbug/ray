"""Implementation of the MPI collective group."""
# try:

# except ImportError:
#     raise ImportError("mpi4py fail to import.")

import logging
import datetime
import time

import ray
import numpy as np
import mpi4py
import mpi4py.MPI as MPI

from ray.util.collective.collective_group import mpi_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.types import AllReduceOptions, BarrierOptions
from ray.util.collective.const import NAMED_ACTOR_STORE_SUFFIX
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

class Rendezvous:
    def __init__(self, group_name):
        if not group_name:
            raise ValueError('Empty meeting point.')
        self._group_name = group_name
        self._store_name = None
        self._store = None

    def meet_at_store(self, timeout=180):
        """Meet at the named actor store."""
        if timeout is not None and timeout < 0:
            raise ValueError("The 'timeout' argument must be nonnegative. "
                             f"Received {timeout}")
        self._store_name = self._group_name + NAMED_ACTOR_STORE_SUFFIX
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

    # def get_mpi_id(self):
    #     if not self._store:
    #         raise ValueError("Rendezvous store is not setup.")
    #     uid = ray.get(self._store.get_id.remote())
    #     return uid

class MPIGroup(BaseGroup):
    def __init__(self, world_size, rank, group_name):
        """Init an MPI collective group."""
        super(MPIGroup, self).__init__(world_size, rank, group_name)

        # default communicator
        self._mpi_comm = MPI.COMM_WORLD

        _rank = self._mpi_comm.rank
        print('mpi world_size: {}, inter_rank: {}, set rank: {} group_name {}'.format(self.size, self.rank, rank, group_name))
        if self.rank == rank:
            self._mpi_comm2 = self._mpi_comm.Create(self._mpi_comm.Get_group())
            self._mpi_comm2 = self._mpi_comm.Create(self._mpi_comm.Get_group())

        self._rendezvous = Rendezvous(self.group_name)
        self._rendezvous.meet_at_store()

        # Setup the mpi uid using the store
        # self._init_mpi_unique_id()

        # assert world_size == self.size

    # def _init_mpi_unique_id(self):
    #     """Init the MPI unique ID required for setting up MPI communicator."""
    #     self._mpi_uid = self._rendezvous.get_mpi_id()

    @classmethod
    def backend(cls):
        return 'mpi'

    @property
    def rank(self):
        return self._mpi_comm.Get_rank()

    @property
    def size(self):
        return self._mpi_comm.Get_size()

    def destroy_group(self):
        """Destroy the group and release the MPI communicators safely."""
        MPI.Finalize()

    def allreduce(self, tensor, allreduce_options=AllReduceOptions()):
        """
        AllReduce a list of tensors following options.

        Args:
            tensor: the tensor to be reduced, each tensor locates on a GPU
            allreduce_options:

        Returns:
        """
        # mpi_util._check_dtype("allreduce", tensor)
        tensor = mpi_util.get_mpi_tensor_obj(tensor)
        dtype = mpi_util.get_mpi_tensor_dtype(tensor)
        op = mpi_util.get_mpi_reduce_op(allreduce_options.reduceOp)

        self._mpi_comm.Allreduce(MPI.IN_PLACE, [tensor, dtype], op)

    def barrier(self, barrier_options=BarrierOptions()):
        self._mpi_comm.Barrier()

    # def _get_mpi_communicator(self):
    #     """
    #     Create or use a cached MPI communicator for the collective task.

    #     """
    #     if not self._mpi_comm:
    #         self._mpi_comm = mpi_util.create_mpi_communicator(
    #             self.world_size, self.mpi_uid, self.rank)
    #     return self._mpi_comm

    @classmethod
    def get_unique_id(cls):
        mpi_comm = MPI.COMM_WORLD
        global_names = mpi_comm.gather(mpi4py.MPI.Get_processor_name())

        if mpi_comm.rank == 0:
            import collections
            name_to_global_ranks = collections.defaultdict(list)
            for global_rank, name in enumerate(global_names):
                name_to_global_ranks[name].append(global_rank)
            for global_ranks in name_to_global_ranks.values():
                global_ranks.sort()

            inter_names = sorted(
                set(global_names), key=lambda name: name_to_global_ranks[name])
            name_to_inter_rank = {
                name: inter_rank
                for inter_rank, name in enumerate(inter_names)
            }
            inter_size = len(inter_names)

            all_ranks = []
            for global_rank, name in enumerate(global_names):
                ranks = name_to_global_ranks[name]
                intra_rank = ranks.index(global_rank)
                intra_size = len(ranks)
                inter_rank = name_to_inter_rank[name]
                all_ranks.append((
                    global_rank, intra_rank, intra_size,
                    inter_rank, inter_size))
            # my_ranks = mpi_comm.scatter(all_ranks)
            return all_ranks
        else:
            raise Exception("Only rank 0 can get_mpi_unique_id")