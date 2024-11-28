import numpy as np
import xarray as xr
from copy import deepcopy

def any_finite(x):
    
    return np.any(np.isfinite(x))

def all_finite(x):
    
    return np.all(np.isfinite(x))

def count_finite(x):
    
    return np.sum(np.isfinite(x))/np.prod(x.shape)

def finitize(x):
    
    return x[np.isfinite(x)]

def symmetrize(mx):
    
    return np.nanmean([mx, np.swapaxes(mx, 0, 1)], axis=0)

def apply_function_expand_dims(data, func, new_dim_name="new_dim", new_coord_values=None):
    """
    Apply a function to each element of numpy.ndarray or an xarray.DataArray, where the function
    returns a vector for each element, expanding the array with a new dimension.
    
    Parameters:
    - data: numpy.ndarray or xarray.DataArray, input data on which the function is applied element-wise.
    - func: callable, function that takes a float and returns a 1D numpy array (vector).
    - new_dim_name: str, name for the new dimension to be added for the function's vector output. Only used for DataArrays.
    - new_coord_values: array-like, custom coordinate values for the new dimension. Only used for DataArrays.
                        If None, defaults to np.arange(length of vector output).
    
    Returns:
    - numpy.ndarray or xarray.DataArray with an added dimension for the vector output of the function.
    """
    values = data if isinstance(data, np.ndarray) else data.values
    vector_length = len(func(values.flat[0]))

    result_array = np.apply_along_axis(lambda x: func(x[0]), -1, values[..., None])
    result_array = result_array.reshape(*data.shape, vector_length)
    if isinstance(data, np.ndarray):
        return result_array

    if new_coord_values is None:
        new_coord_values = np.arange(vector_length)
    elif len(new_coord_values) != vector_length:
        raise ValueError("Length of new_coord_values must match the length of the vector returned by func.")
    
    result = xr.DataArray(
        result_array.reshape(*data.shape, vector_length),
        dims=(*data.dims, new_dim_name),
        coords={**data.coords, new_dim_name: new_coord_values}
    )
    return result

def is_square(matrix, require=False):
    cond = len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    if require and not cond:
        raise ValueError(f'Matrix of shape {matrix.shape} is not square!')
    return cond

def is_symmetric(matrix, require=False):
    # running into floating point issues just comparing matrix and matrix.T
    cond = np.isclose(matrix - matrix.T, np.zeros_like(matrix)).all()
    if require and not cond:
        raise ValueError(f'Matrix is not symmetric!')
    return cond

def is_positive_definite(matrix, require=False):
    """Checks if a NumPy matrix is positive definite.
    """
    is_square(matrix, require=True)
    try:
        np.linalg.cholesky(matrix)
        cond = True
    except np.linalg.LinAlgError:
        cond = False
    if require and not cond:
        raise ValueError(f'Matrix is not positive definite!')
    return cond

def strict_triu(arr):
    arr = np.triu(arr)
    np.fill_diagonal(arr, 0)
    return arr

def upper_tri_values(matrix, include_diagonal=True):
  """Extracts the upper triangular values of a matrix.

  Args:
    matrix: A NumPy matrix.
    include_diagonal: Whether to include the main diagonal. 
                       Defaults to True.

  Returns:
    A 1D NumPy array containing the upper triangular values.
  """
  if include_diagonal:
    return matrix[np.triu_indices(matrix.shape[0])]
  else:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

def off_diagonal_values(matrix):
  """Extracts the off-diagonal values of a matrix.

  Args:
    matrix: A NumPy matrix.

  Returns:
    A 1D NumPy array containing the off-diagonal values.
  """
  mask = ~np.eye(matrix.shape[0], dtype=bool)  # Create a mask for off-diagonal elements
  return matrix[mask]


def apply_indexing(arr, indices, axis):
    """Applies indexing operation to a NumPy array along a specified axis.

    Args:
        arr: The NumPy array to be indexed.
        indices: A 1D array of indices to apply.
        axis: The axis along which to apply the indexing.

    Returns:
        The indexed NumPy array.
    """

    all_indices = [slice(None)] * arr.ndim  # Create slices for all dimensions
    all_indices[axis] = indices  # Replace the slice for the target axis with the indices
    return arr[tuple(all_indices)] 


def sort_array_across_order(arr, order, axis=0, invert_sort=False):
    if isinstance(order, list):
        order = np.array(order)
    assert len(order.shape) == 1
    sort_idx = np.argsort(order)
    # print(sort_idx)
    if invert_sort:
        sort_idx = np.argsort(sort_idx)
        # print(sort_idx)
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        arr = apply_indexing(arr, sort_idx, ax)
    return arr

