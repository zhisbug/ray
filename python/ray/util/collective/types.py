"""Types conversion between different backends."""
from enum import Enum
from datetime import timedelta

_NUMPY_AVAILABLE = True
_TORCH_AVAILABLE = True
_CUPY_AVAILABLE = True

try:
    import numpy as np
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import torch as th
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import cupy as cp
except ImportError:
    _CUPY_AVAILABLE = False


def numpy_available():
    return _NUMPY_AVAILABLE


def cupy_available():
    return _CUPY_AVAILABLE


def torch_available():
    return _TORCH_AVAILABLE


class Backend(object):
    """A class to represent different backends, in case the user string is too error-prone."""
    NCCL = 'nccl'
    MPI = 'mpi'
    UNRECOGNIZED = 'unrecognized'

    def __new__(cls, name: str):
        backend = getattr(Backend, name.upper(), Backend.UNRECOGNIZED)
        if backend == Backend.UNRECOGNIZED:
            raise ValueError("Unrecognized backend: '{}'"
                             "Only NCCL is supported".format(name))
        if backend == Backend.MPI:
            raise NotImplementedError()
        return backend


# TODO(Hao): extend this to support more MPI types
class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3


unset_timeout = timedelta(milliseconds=-1)


class AllReduceOptions:
    reduceOp = ReduceOp.SUM
    timeout = unset_timeout


class BarrierOptions:
    timeout = unset_timeout