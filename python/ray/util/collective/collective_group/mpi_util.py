"""Code to wrap some MPI API calls."""
import mpi4py
import numpy


from ray.util.collective.types import ReduceOp, torch_available

MPI_REDUCE_OP_MAP = {
    ReduceOp.SUM: mpi4py.MPI.SUM,
    ReduceOp.PRODUCT: mpi4py.MPI.PROD,
    ReduceOp.MIN: mpi4py.MPI.MIN,
    ReduceOp.MAX: mpi4py.MPI.MAX,
}

NUMPY_MPI_DTYPE_MAP = {
    # see the definition of mpi4py.MPI._typedict (in mpi4py/MPI/typemap.pxi)
    numpy.dtype(numpy.bool): mpi4py.MPI._typedict['?'],
    numpy.dtype(numpy.int32): mpi4py.MPI._typedict['i'],
    numpy.dtype(numpy.int64): mpi4py.MPI._typedict['l'],
    numpy.dtype(numpy.float16): mpi4py.MPI._typedict['f'],
    numpy.dtype(numpy.float32): mpi4py.MPI._typedict['f'],
    numpy.dtype(numpy.float64): mpi4py.MPI._typedict['d'],
}

if torch_available():
    import torch
    TORCH_MPI_DTYPE_MAP = {
        torch.bool: mpi4py.MPI._typedict['?'],
        torch.long: mpi4py.MPI._typedict['l'],
        torch.float16: mpi4py.MPI._typedict['f'],
        torch.float32: mpi4py.MPI._typedict['f'],
        torch.float64: mpi4py.MPI._typedict['d'],
    }


def _check_dtype(caller, msg):
    dtype = msg.dtype
    if dtype not in NUMPY_MPI_DTYPE_MAP.keys() \
        and dtype not in TORCH_MPI_DTYPE_MAP.keys():
        raise TypeError(
            '{} does not support dtype {}'.format(caller, dtype))

def get_mpi_reduce_op(reduce_op):
    """
    Map the reduce op to mpi reduce op type.

    Returns:
        MPI_op (mpi4py.MPI.Op)
    """
    if reduce_op not in MPI_REDUCE_OP_MAP:
        raise RuntimeError('MPI does not support ReduceOp: {}'.format(reduce_op))
    return MPI_REDUCE_OP_MAP[reduce_op]

def get_mpi_tensor_dtype(tensor):
    """Return the corresponded MPI dtype given a tensor."""
    if isinstance(tensor, numpy.ndarray):
        return NUMPY_MPI_DTYPE_MAP[tensor.dtype]
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            return TORCH_MPI_DTYPE_MAP[tensor.dtype]
    raise ValueError('Unsupported tensor type')

def get_mpi_tensor_obj(tensor):
    """Return tensor object."""
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    if isinstance(tensor, numpy.ndarray):
        return tensor.data
    raise ValueError('Unsupported tensor type')
