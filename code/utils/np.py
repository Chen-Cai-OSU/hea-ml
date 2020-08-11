# Created at 2020-08-10
# Summary:

import numpy as np
import scipy
import torch

np.random.seed(42)


def isndarray(arr):
    return isinstance(arr, np.ndarray)


def one_hot(label, nr_classes, dtype='float32'):
    if isinstance(label, int) or (isndarray(label) and len(label.shape) == 0):
        out = np.zeros(nr_classes, dtype=dtype)
        out[int(label)] = 1
        return out

    assert len(label.shape) == 1
    nr_labels = label.shape[0]
    out = np.zeros((nr_labels, nr_classes), dtype=dtype)
    out[np.arange(nr_labels), label] = 1
    return out


def index_of_firstx(arr, x=1, not_found=-1):
    """
        http://bit.ly/2TvAGoy
        For array of shape (n, d), find the indices of first x in each row.
        return array of shape (n,). If there is not x in certain row, return -1
    """
    mask = arr == x
    ans = np.where(mask.any(1), mask.argmax(1), not_found)
    ans = ans.reshape((len(ans), 1))
    return ans


def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)
    try:
        arr = tsr.numpy()
    except:
        arr = tsr.detach().numpy()
    assert isinstance(arr, np.ndarray)
    return arr


def totsr(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    else:
        assert isinstance(arr, np.ndarray)
        return torch.Tensor(arr)


def norm(x, ord=None):
    """ compute the norm a np.array/tensor. by default 2-norm. """
    assert isinstance(x, (np.ndarray, torch.Tensor))
    x = tonp(x)
    return scipy.linalg.norm(x, ord=ord)


def np2set(x):
    assert isinstance(x, np.ndarray)
    return set(np.unique(x))


def rm_zerocol(data, print_flag=False):
    """ remove zero columns """
    x = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)

    if print_flag:
        print(f'the shape before/after removing zero columns is {np.shape(data)}/{np.shape(x)}')

    return x


def non_zero(x):
    """ get the non-zero values of an array """
    assert isinstance(x, np.ndarray)
    return x.ravel()[np.flatnonzero(x)]


if __name__ == '__main__':
    label = np.array([1, 2, 3])
    print(one_hot(label, 4))
