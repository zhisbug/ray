import logging
import datetime
import time

import ray
import cupy
from cupy.cuda.nccl import groupStart, groupEnd
from cupy.cuda import Device, Event, Stream, runtime, get_current_stream
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group \
    import BaseGroup
from ray.util.collective.types import AllReduceOptions, \
    BarrierOptions, Backend
from ray.util.collective.const import get_nccl_store_name

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
# TODO(Hao):
# (1) stream management, instead of using the default stream,
#     using a dedicate stream
# (2) communicator management and support num_gpus > 2 per actor.


class Rendezvous:
    """
    A rendezvous class for different actor/task processes to meet.

    To initialize an NCCL collective communication group, different
    actors/tasks spawned in Ray in a collective group needs to meet
    each other to synchronize the NCCLUniqueID. This class guarantees
    they meet via the NCCLUniqueIDStore, initialized on the rank=0
    process.

    Args:
        group_name (str): the unique user-specified group name.
    """

    def __init__(self, group_name):
        if not group_name:
            raise ValueError("Invalid group name.")
        self._group_name = group_name
        self._store_name = None
        self._store = None

    def meet(self, timeout_s=180):
        """
        Meet at the named actor store.

        Args:
            timeout_s: timeout in seconds.

        Return:
            None
        """
        if timeout_s <= 0:
            raise ValueError("The 'timeout' argument must be positive. "
                             "Got '{}'.".format(timeout_s))
        self._store_name = get_nccl_store_name(self._group_name)
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            try:
                logger.debug("Trying to meet at the store '{}'".format(
                    self._store_name))
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

    def get_nccl_id(self, timeout_s=180):
        """
        Get the NCCLUniqueID from the store through Ray.

        Args:
            timeout_s: timeout in seconds.
        Return:
            str: the NCCLUniqueID if successful.
        """
        if not self._store:
            raise ValueError("Rendezvous store is not setup.")
        uid = None
        timeout_delta = datetime.timedelta(seconds=timeout_s)
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
                "Unable to get the NCCLUniqueID from the store.")
        return uid


class NCCLGroup(BaseGroup):
    def __init__(self, world_size, rank, group_name):
        """Init an NCCL collective group."""
        super(NCCLGroup, self).__init__(world_size, rank, group_name)
        #self._nccl_uid = None

        # TODO(Hao): change this to a be a cache
        #self._nccl_comm = None

        if nccl_util.get_nccl_build_version() < 2000:
            raise RuntimeError("NCCL in Ray requires NCCL >= 2.0.")
        # TODO(Hao): check version here
        if nccl_util.get_nccl_runtime_version() < 2704:
            logger.warning("NCCL send/recv calls requires NCCL>=2.7.4")

        # Setup a tensor for barrier calls
        self._barrier_tensor = cupy.array([1])

        self._dev_comm_map = dict()
        self._dev_streams_map = dict()

    def destroy_group(self):
        """
        Destroy the group and release the NCCL communicators safely.

        """
        if len(self._dev_comm_map.keys()) > 0:
            self.barrier()
            # destroy the streams and  communicator
            for _, stream in self._dev_streams_map.items():
                runtime.streamDestroy(stream)
            
            for _, comms in self._dev_comm_map.items():
                [comm.destroy() for c in comms]

            self._barrier_tensor = None
            self._dev_comm_map = None
            self._dev_streams_map = None
        super(NCCLGroup, self).destroy_group()

    @classmethod
    def backend(cls):
        return Backend.NCCL

    def allreduce(self, tensor, allreduce_options=AllReduceOptions()):
        """
        AllReduce a list of tensors following options.

        Args:
            tensor: the tensor to be reduced, each tensor locates on a GPU
            allreduce_options:

        Returns:
        """
        nccl_util.check_collective_input(tensor) 
        devices = nccl_util.get_devices(tensor)
        key = nccl_util.get_key_from_devices(devices)
        # obtain the communicator
        # obtain the stream: using default stream by now
        # TODO(Hao): implement a simple stream manager here
        comms = self._get_nccl_communicator(devices)
        reduce_op = nccl_util.get_nccl_reduce_op(allreduce_options.reduceOp)
       
        # First wait for current tensor allocation stream
        streams = self._dev_streams_map[key]
        self._sync_streams()
        # for non-blocking calls of all-reduce
        groupStart()
        for i in range(len(tensor)):
            dtype = nccl_util.get_nccl_tensor_dtype(tensor[i])
            ptr = nccl_util.get_tensor_ptr(tensor[i])
            n_elems = nccl_util.get_tensor_n_elements(tensor[i])
            # in-place allreduce
            comms[i].allReduce(ptr, ptr, n_elems, dtype, reduce_op, streams[i].ptr)
        groupEnd()

    def barrier(self, barrier_options=BarrierOptions()):
        """
        Blocks until all processes reach this barrier.

        Args:
            barrier_options:

        Returns:
        """
        self.allreduce(self._barrier_tensor)

    def _get_nccl_communicator(self, devices):
        """
        Create or use a cached NCCL communicator for the collective task.

        """
        # TODO(Hao): later change this to use device keys and query from cache.
        # TODO(Hao): implement a thin wrapper
        # try to find from cache
        key = nccl_util.get_key_from_devices(devices)
        if key in self._dev_comm_map.keys():
            return self._dev_comm_map[key]
        else: # create a new one and cache
            _group_name = self.group_name + key
            if self.rank == 0:
                uid = nccl_util.get_nccl_unique_id()
                _store_name = get_nccl_store_name(_group_name)
                from ray.util.collective.util import NCCLUniqueIDStore
                store = NCCLUniqueIDStore.options(
                    name=_store_name, lifetime="detached").remote(_store_name)
                ray.wait([store.set_id.remote(uid)])

            rendezvous = Rendezvous(_group_name)
            rendezvous.meet()
            nccl_uid = rendezvous.get_nccl_id()
            _world_size = len(devices) * self.world_size
            comms = []
            
            nccl_streams = []
            # for non-blocking communicator creation
            groupStart()
            for i in range(len(devices)):
                _rank = self.rank * len(devices) + i
                from cupy.cuda import Device
                with Device(devices[i]):
                    comm = nccl_util.create_nccl_communicator(
                                _world_size, nccl_uid, _rank)
                    stream = Stream(non_blocking=True)
                    logger.debug(f"{stream}")
                comms.append(comm)
                nccl_streams.append(stream)
            groupEnd()

        # cache the result
        # FIXME: Consider whether to add a lock here or not, I feel like pytorch
        # needs to handle this because they are relatively lower level, we shouldnt
        # need to worry about this. if so,eed to figure out how to add Lock, threading? Asyncio?
        self._dev_comm_map[key] = comms
        self._dev_streams_map[key] = nccl_streams
        return comms
    
    def _sync_streams(self):
        """Let Nccl streams wait for current streams for every device."""
        #FIXME: This behavior is different from nccl document. It seems like
        # cupy allocate tensors on null streams.
        cupy.cuda.Stream.null.synchronize()
    
    # def _collective_call(self, *args):
    #     """Private method to encapsulate all collective calls"""
    #     pass
