# Created at 2020-08-10
# Summary:
from itertools import combinations

import numpy as np


def tolist(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()


def expand(lis):
    """ expand a nested list into a non-nest one """
    res = []
    for l in lis:
        if isinstance(l, list):
            res += expand(l)
        else:
            res.append(l)
    return res


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r))) # https://bit.ly/345tVh7
    return list(combinations(arr, r))


# Driver Function
def sublist(a, indices):
    assert isinstance(a, list)
    return [a[index] for index in indices]


def isflatlist(lis):
    return not any([isinstance(i, list) for i in lis])


if __name__ == '__main__':
    a = list(range(10))
    indices = [1, 3, 7]
    print(sublist(a, indices))

