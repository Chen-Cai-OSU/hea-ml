# Created at 2020-08-10
# Summary:
import random
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from pprint import pprint

import torch.nn as nn
import yaml

from code.utils.format import banner, red
from code.utils.probe import summary

random.seed(42)


def dict_product(d, shuffle=False):
    """
    :param d: d = {'x': [1,2,3], 'y':['a', 'b', 'c']}
    :param shuffle:
    :return: a list of dict

    >>> d = {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
    >>> res = dict_product(d)
    >>> assert len(res) == 9

    """

    assert isinstance(d, dict)
    keys = d.keys()
    results = []

    for element in product(*d.values()):
        results.append(dict(zip(keys, element)))

    if shuffle: random.shuffle(results)
    print(f'There are {red(len(results))} combinations.')
    return results


def my_function(a, b):
    """
    >>> my_function(2, 3)
    6
    >>> my_function('a', 3)
    'aaa'
    """
    return a * b


def dict2arg(d, verbose=0, sacred=False):
    """
    :param d: {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    :param sacred: if true. with sacred arg version. with n_epoch=300 bs=32
    :return: --scheduler --n_epoch 300 --bs 32 --n_data 1000

    >>> d =  {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    >>> assert dict2arg(d) == '--scheduler  --bs 32 --n_data 10 --n_epoch 300'
    """

    d = OrderedDict(sorted(d.items(), reverse=False))
    copy_d = deepcopy(d)
    if verbose: print(d)
    arg = 'with ' if sacred else ''

    for k, v in OrderedDict(d).items():
        if v is True:
            arg = arg + f"'{k}={v}' " if sacred else arg + f'--{str(k)} '
            copy_d.pop(k)
        elif v is False:  # do nothing
            arg = arg + f"'{k}={v}' " if sacred else arg
            copy_d.pop(k)
        else:
            pass

    copy_d = OrderedDict(copy_d)
    for k, v in copy_d.items():
        if sacred:
            arg += f"'{k}={v}' "
        else:
            arg += f' --{k} {v}'

    if verbose:
        print(arg)
        print('-' * 10)

    return arg


def dict2name(d, flat=True, ):
    """
    :param d: {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    :return: bs_32_n_data_10_n_epoch_300_scheduler_True
    """
    assert isinstance(d, dict)
    keys = list(d.keys())
    keys.sort()
    name = ''

    if flat:
        for k in keys:
            name += f'{k}_{d[k]}_'
        return name[:-1]
    else:
        for k in keys:
            name += f'{k}_{d[k]}/'
        return name


def val_from_name(name, key):
    """ from name a_1_b_2 get the value corresponding key """
    # todo: only works when key doesn't contain _
    assert '_' in name
    name = name.split('_')
    assert key in name, f'key {key} not in {name}'
    key_idx = name.index(key)
    val_idx = key_idx + 1
    return float(name[val_idx])


class subset_dict():
    " get the subset of origianl dict"

    def __init__(self, d):
        assert isinstance(d, dict)
        self.d = d
        self.keys = d.keys()

    def include(self, keys):
        assert isinstance(keys, list)
        _d = dict()
        for key in keys:
            try:
                _d[key] = self.d[key]
            except KeyError:
                exit(f'{key} in not the key of {self.d}')
        return _d

    def exclude(self, keys):
        assert isinstance(keys, list)
        _d = deepcopy(self.d)
        for key in self.keys:
            if key in keys:
                _d.pop(key)
        return _d


def load_configs(f):
    """ load configs from yaml file and convert to a dict
        Can be used for hyperparameter search.
    """
    with open(f, 'r', encoding='utf-8') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.FullLoader)
    cfg_dic = {}

    for k, v in cfg.items():
        cfg_dic[k] = v['values']

    return cfg_dic

import os.path as osp
def curdir():
    return osp.dirname(osp.realpath(__file__))

def load_deepset_yaml(id=0):
    f = osp.join(curdir(), '..', 'deepset.yaml')
    cfg_dic = load_configs(f)

    cfg_dic['extractor_nl'] = [nn.Identity]  # [nn.ReLU, nn.ELU, nn.Identity] #
    cfg_dic['regressor_nl'] = [nn.ReLU, nn.ELU]

    del cfg_dic['device']
    del cfg_dic['verbose']

    # remark: used this to quickly test an idea
    # cfg_dic['extractor_nl'] = [nn.ELU]# [nn.ReLU, nn.ELU] #
    # cfg_dic['in_features'] = [94]
    # cfg_dic['set_features'] = [10, 30, 50]
    # cfg_dic['regressor_hidden_dim'] = [[100, 50]]# [[10], [30], [50]]

    # good for 4_non_equal
    cfg_dic['regressor_nl'] = [nn.ReLU, nn.ELU]
    cfg_dic['extractor_nl'] = [nn.Identity, nn.ELU, nn.ReLU]
    cfg_dic['extractor_hidden_dim'] = [[41]]  # [[94], [94, 50], [50, 50]]
    cfg_dic['in_features'] = [41]
    cfg_dic['bn'] = [True, False]
    cfg_dic['set_features'] = [41, ]

    model_params = dict_product(cfg_dic, shuffle=True)
    assert id < len(model_params), f'{id} larger than num of combinations({len(model_params)})'
    opt_param = {'batch_size': 32, 'lr': 0.001, 'verbose': 0, 'max_epochs': 300}
    return model_params[id], opt_param


if __name__ == '__main__':
    d = {'x': 1, 'y': 2}

    for f in [True, False]:
        name = dict2name(d, flat=f)
        print(name)

    exit()
    all = load_deepset_yaml()
    summary(all, 'all')
    banner()
    pprint(all)

    exit()

    d = {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
    res = dict_product(d, shuffle=True)
    pprint(res)

    exit()
    name = 'ac_1_b_2'
    print(val_from_name(name, 'a'))

    exit()
    d = {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    print(dict2name(d))

    print(subset_dict(d).include(['n_epoch', 'bs']))
    print(subset_dict(d).exclude(['n_epoch', 'bs']))

    exit()

    print(sorted(d.items(), reverse=False))
    print(dict2arg(d))
    pass
