# Created at 2020-08-10
# Summary:
""" get the shape of everything """

import collections
import inspect
import sys
from functools import partial

import numpy as np
import pandas as pd
import scipy
import torch
from code.utils.format import banner, red, pf
from code.utils.np import tonp, np2set

nan1 = 0.12345


def stats(x, precision=2, verbose=True, var_name='None'):
    """
    print the stats of a (np.array, list, pt.Tensor)

    :param x:
    :param precision:
    :param verbose:
    :return:
    """
    if isinstance(x, torch.Tensor): x = tonp(x)
    assert isinstance(x, (list, np.ndarray)), 'stats only take list or numpy array'

    ave_ = np.mean(x)
    median_ = np.median(x)
    max_ = np.max(x)
    min_ = np.min(x)
    std_ = np.std(x)
    pf_ = partial(pf, precision=precision)

    if verbose:
        ave_, min_, max_, median_, std_ = list(map(pf_, [ave_, min_, max_, median_, std_]))
        line = '{:>25}: {:>5}(mean) {:>5}(min) {:>5}(max) {:>5}(median) {:>5}(std)'.format(var_name, ave_, min_, max_,
                                                                                           median_, std_)
        print(line)

    return list(map(pf_, [ave_, min_, max_, median_, std_]))


# from torch_geometric.data.data import Data
def summary(x, name='x', terminate=False,
            skip=False, delimiter=None, precision=3,
            exit=False, highlight=False):
    if highlight:
        name = red(name)

    if skip:
        print('', end='')
        return ''

    if isinstance(x, list):
        print(f'{name}: a list of length {len(x)}')

        if len(x) < 6:
            for _x in x:
                summary(_x)

    elif isinstance(x, scipy.sparse.csc.csc_matrix):
        min_, max_ = x.min(), x.max()
        mean_ = x.mean()

        std1 = np.std(tonp(x))
        x_copy = x.copy()
        x_copy.data **= 2
        std2 = x_copy.mean() - (x.mean() ** 2)  # todo: std1 and std2 are different
        pf_ = partial(pf, precision=precision)
        mean_, min_, max_, std1, std2 = list(map(pf_, [mean_, min_, max_, std1, std2]))

        line0 = '{:>10}: csc_matrix ({}) of shape {:>8}'.format(name, str(x.dtype), str(x.shape))
        line0 = line0 + ' ' * max(5, (45 - len(line0)))
        # line0 += 'Nan ratio: {:>8}.'.format(nan_ratio(x_))
        line1 = '  {:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(std1) {:>8}(std2) {:>8}(unique) ' \
            .format(mean_, min_, max_, std1, std2, -1)
        line = line0 + line1
        print(line)

    elif isinstance(x, (np.ndarray,)):
        if x.size > 232960 * 10:
            return
        x_ = tonp(x)
        ave_ = np.mean(x_)
        median_ = np.median(x_)
        max_ = np.max(x_)
        min_ = np.min(x_)
        std_ = np.std(x_)
        unique_ = len(np.unique(x_))
        pf_ = partial(pf, precision=precision)
        ave_, min_, max_, median_, std_, unique_ = list(map(pf_, [ave_, min_, max_, median_, std_, unique_]))

        line0 = '{:>10}: array ({}) of shape {:>8}'.format(name, str(x.dtype), str(x.shape))
        line0 = line0 + ' ' * max(5, (45 - len(line0)))
        line0 += 'Nan ratio: {:>8}.'.format(nan_ratio(x_))
        line1 = '  {:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(median) {:>8}(std) {:>8}(unique) '.format(ave_, min_, max_,
                                                                                                     median_, std_,
                                                                                                     unique_)
        line = line0 + line1
        if np2set(x_) <= set([-1, 0, 1]):
            ratio1 = np.sum(x_ == 1) / float(x_.size)
            ratio0 = np.sum(x_ == 0) / float(x_.size)
            line += '|| {:>8}(1 ratio) {:>8}(0 ratio)'.format(pf(ratio1, 3), pf(ratio0, 3))

        if nan1 in x_:
            nan_cnt = np.sum(x_ == nan1)
            line += f'nan_cnt {nan_cnt}'

        # f'{name}: array of shape {x.shape}.'
        print(line)
        # print(f'{name}: a np.array of shape {x.shape}. nan ratio: {nan_ratio(x)}. ' + line)

    elif isinstance(x, (torch.Tensor)):
        if x.numel() > 232965 * 10:
            return
        x_ = tonp(x)
        if len(x_) == 0:
            print(f'{name}: zero length np.array')
        else:
            ave_ = np.mean(x_)
            median_ = np.median(x_)
            max_ = np.max(x_)
            min_ = np.min(x_)
            std_ = np.std(x_)
            unique_ = len(np.unique(x_))

            pf_ = partial(pf, precision=2)
            ave_, min_, max_, median_, std_, unique_ = list(map(pf_, [ave_, min_, max_, median_, std_, unique_]))
            line = '{:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(median) {:>8}(std) {:>8}(unique)'.format(ave_, min_, max_,
                                                                                                     median_, std_,
                                                                                                     unique_)

            print(
                '{:35}'.format(name) + '{:20}'.format(str(x.data.type())[6:]) + '{:15}'.format(
                    str(x.size())[11:-1]) + line)
        # print(line)
        # print(f'{name}: a Tensor ({x.data.type()}) of shape {x.size()}')

    elif isinstance(x, tuple):
        print(f'{name}: a tuple of shape {len(x)}')
        if len(x) < 6:
            for ele in x:
                summary(ele, name='ele')

    elif isinstance(x, (dict, collections.defaultdict)):
        print(f'summarize a dict {name} of len {len(x)}')
        for k, v in x.items():
            # print(f'key is {k}')
            summary(v, name=k)

    elif isinstance(x, pd.DataFrame):
        from collections import OrderedDict

        dataType_dict = OrderedDict(x.dtypes)
        banner(text=f'start summarize a df ({name}) of shape {x.shape}', ch='-')
        print('df info')
        print(x.info())
        print('\n')

        print('head of df:')
        # print(tabulate(x, headers='firstrow'))
        print(x.head())
        print('\n')

        try:
            print('continuous feats of Dataframe:')
            cont_x = x.describe().T
            print(cont_x)
            print(cont_x.shape)
            print('\n')
        except ValueError:
            print('x.describe().T raise ValueError')

        try:
            print('non-cont\' feats (object type) of Dataframe:')
            non_cont = x.describe(include=[object]).T
            print(non_cont)
            print(non_cont.shape)
        except ValueError:
            print('x.describe(include=[object]).T raise ValueError')

        banner(text=f'finish summarize a df ({name}) of shape {x.shape}', ch='-')

    elif isinstance(x, (int, float)):
        print(f'{name}(float): {x}')

    elif isinstance(x, str):
        print(f'{name}(str): {x}')

    else:
        print(f'{x}: \t\t {type(x)}')
        if terminate:
            exit(f'NotImplementedError for input {type(x)}')
        else:
            pass

    if delimiter is not None:
        assert isinstance(delimiter, str)
        print(delimiter)

    if exit:
        sys.exit()


def nan_ratio(x):
    """ http://bit.ly/2PL7yaP
    """
    assert isinstance(x, np.ndarray)
    try:
        return np.count_nonzero(np.isnan(x)) / x.size
    except TypeError:
        return '-1 (TypeError)'


if __name__ == '__main__':
    x = ['1', 'abc', 'xyz']
    summary(np.array(x))
