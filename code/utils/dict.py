# Created at 2020-08-10
# Summary:

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import torch


def filter_dict(d, keys, inclde=True):
    """ filter dict d by keys. """
    assert isinstance(keys, list)
    assert isinstance(d, dict)
    d_filter = dict()

    if inclde:
        for k, v in d.items():
            if k in keys:
                d_filter[k] = v
    else:
        for k, v in d.items():
            if k not in keys:
                d_filter[k] = v

    return d_filter


def subset_dict(d, start=0, end=5):
    """ for a dict where values are all lists of same length n.
        subset from start to end
    """

    n = len(list(d.values())[0])
    for k, v in d.items():
        assert isinstance(v, (list, np.ndarray)), f'Expect value of dict to be list/array, not {type(v)}'
        if isinstance(v, list): v = np.array(v)
        if end > n:
            print(f'Warning: end {end} not larger than {n}. Truncated to {n}')
            end = n
        d[k] = v[start:end]
    return d


def viz_dict(result):
    """ visualize a dict where keys are list/np.array
        Usually such dict is train history of some model.
    """

    for k, v in result.items():
        assert isinstance(v, (list, np.ndarray)), f'Expect value of dict to be list/array, not {type(v)}'
        plt.plot(v, label=k)
    title = ''

    plt.legend()
    plt.title("\n".join(wrap(str(title))))
    plt.show()


def update_dict(d1, d2):
    # use d1 to update d2, return updated d2.
    # keys of d1 has to be a subset of keys of d2.
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert set(d1.keys()) <= set(d2.keys()), 'Keys of d1 has to be a subset of keys of d2.'
    for k, v in d1.items():
        d2[k] = v
    return d2


def merge_two_dicts(x, y):
    assert isinstance(x, dict) and isinstance(y, dict), f'x is {type(x)}. y is {type(y)}'
    assert bool(set(x.keys()) & set(y.keys())) == False, \
        f'Expect two dict has no common keys. Found keys {x.keys()} and keys {y.keys()} intersect.'
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def dic2tsr(d, dev='cuda'):
    """ given a dict where key (size N) are consecutive numbers and values are also numbers (at most n),
        convert it into a tensor of size (N) where index is the key value is the value of d.
    """
    N = len(d)
    assert N == max(d.keys()) + 1, f'keys ({N}) are not consecutive. Max key is {max(d.keys)}'
    tsr = [0] * N
    for k in d:
        tsr[k] = d[k]
    return torch.tensor(tsr).to(dev)


if __name__ == '__main__':
    d = {0: 1, 1: 3, 2: 5}
    tsr = dic2tsr(d)
    print(tsr)
    exit()
    d1 = {'a': 1}
    d2 = {'a': 2, 'b': 2}
    print(update_dict(d1, d2))

