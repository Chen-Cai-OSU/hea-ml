""" simple baseline for hea
    some old functions can be found in archive/fun.py
"""

import numpy as np
import torch
from sklearn.preprocessing import normalize

from code.baselines import clf
from code.utils.config_util import load_deepset_yaml
from code.utils.data_util import data_util
from code.utils.dir import hea_sys_dir
from code.utils.format import banner, timestamp, red
from code.utils.list import expand
from code.utils.np import rm_zerocol, totsr
from code.utils.probe import summary
from code.utils.screen import random_heas


class Feat():
    def __init__(self):
        pass

    def hea_feat(self, hea, D, emb='one_hot', ind=False, norm_w=False):
        """
        :param hea: str like CrNbCuFe
        :param D: instance of data_util()
        :param emb: one_hot/neural
        :param ind: return feats for individual atom
        :return: hea feat
        """
        assert emb in ['one_hot', 'neural', 'naive']

        elements = D.get_compositions(hea)
        weights = D.get_weights(hea)
        if norm_w:
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

        assert len(elements) in [4, 5], f'num of elements {elements} for {hea} is not 4/5.'
        feat = D.get_feat(-1, emb=emb)
        ind_feats = []
        for i, ele in enumerate(elements):
            atom_num = D.get_atom_num(ele)
            tmp_feat = D.get_feat(atom_num, emb=emb)  # feat for individual atom
            feat += tmp_feat * weights[i]  # weighted sum of individual atom feat
            ind_feats.append(tmp_feat)  # concatenate atom feat. used for deepset.

        try:
            cell_feat = D.get_cell_feat(hea)
            lattice_feat = D.get_lattice_constant(hea)
        except:  # mainly used for load_hea.py (does not contain cell_feat)
            cell_feat = np.array([[1]])
            lattice_feat = np.array([[2.9]])

        if ind:
            res = np.concatenate(ind_feats, axis=0)  # no cell_feat, array of shape (4, 64)
            cell_feat = np.concatenate([[cell_feat[0]]] * len(elements), axis=0)  # array of shape (4, 1)
            lattice_feat = np.concatenate([[lattice_feat[0]]] * len(elements), axis=0)  # array of shape (4, 1)
            res = np.concatenate([res, cell_feat, lattice_feat], axis=1)
            return res
        else:
            feat = np.concatenate([feat, cell_feat, lattice_feat], axis=1)  # add cell_feat
            return feat


def getxy(ylog=False, emb='one_hot', bm=True, weight=False, ind=False, dir='Processed', prop='bulk', screen=False,
          norm_w=False):
    """
    :param ylog: take log or not
    :param emb: one_hot, naive, neural
    :param bm: True by default
    :param ind: return features for individual atom
    :param dir: Processed/4_ele_fully_random_result
    :param prop: bulk/lattice_constant
    :param screen: if true, generate random 4 element heas
    :return:
    """

    assert dir in ['Processed', '4_ele_fully_random_result', 'NbTaTiVZr_ratio_result', '4_ele_fully_random_verify',
                   'Processed_with_EC']
    dir = f'{hea_sys_dir()}{dir}'  #
    print(dir)
    D = data_util(log=ylog, bm=bm, dir=dir)
    heas = D.heas if not screen else random_heas(n=100)

    F = Feat()

    print(f'first 5 heas: {heas[:5]}')

    x, y, weights = [], [], []
    for i, hea in enumerate(heas):
        feat = F.hea_feat(hea, D, emb=emb, ind=ind, norm_w=norm_w)
        w = D.get_weights(hea)
        if norm_w:
            total_w = sum(w)
            w = [w_ / total_w for w_ in w]

        if i % 300 == 0: summary(feat, name=i)
        y_ = D.y(hea, prop=[prop]) if not screen else np.array([[111]])
        if prop == 'bulk' and expand(y_)[0] < 50:  # used for check prop
            print(hea, y_)
            exit('Expect large bulk')

        x.append(feat)
        weights.append(w)
        y.append(expand(y_))

    if weight:
        return x, weights, y, heas
    else:
        return x, y, heas


def query(res, metal='W'):
    banner(f'query {metal}')
    for k, v in res.items():
        if metal in k:
            if v[metal] == max(v.values()): print(red(v[metal]))
            print(k, v)


def load_data(args, exit=False, screen=False):
    if args.mat == '4_equal':
        dir = 'Processed'
    elif args.mat == '4_non_equal':
        dir = '4_ele_fully_random_result'
    elif args.mat == '5_non_equal':
        dir = 'NbTaTiVZr_ratio_result'
    elif args.mat == 'ec':
        dir = 'Processed_with_EC'
    else:
        raise Exception(f'Not implemented for {args.mat}')

    kw = {'ylog': args.log, 'emb': args.emb, 'ind': False, 'bm': True, "dir": dir, 'prop': args.prop, 'screen': screen,
          'norm_w': args.norm_w}

    if args.method not in ['deepset', 'weight']:
        x_list, y_list, labels_list = getxy(**kw)

        x = np.concatenate(x_list, axis=0)
        x = rm_zerocol(x)
        y = np.array(expand(y_list)).ravel()
    else:
        if args.method == 'deepset':
            kw['ind'] = True
            kw['weight'] = True

            x_list, w_list, y_list, labels_list = getxy(**kw)

            w = [torch.Tensor(w_) for w_ in w_list]
            x = [torch.Tensor(x_) for x_ in x_list]
            y = [torch.Tensor(y_) for y_ in y_list]
            x, w, y = np.stack(x, axis=0), np.stack(w, axis=0), np.concatenate(y, axis=0)

            x = torch.tensor(x)
            w = torch.tensor(w.reshape((len(labels_list), -1, 1)))
            x = torch.mul(x, w)

            x = x[:, :, torch.tensor(x).sum(dim=[0, 1]) != 0]  # remove all zeros (779, 5, 94) --> (779, 5, 21)
        elif args.method == 'weight':  # hack for get the weights
            kw['weight'] = True
            _, weights, y_list, labels_list = getxy(**kw)
            x = np.array(weights)
            prefix = 'norm'
            x = normalize(x, norm='l1', axis=1)

            y = np.array(expand(y_list)).reshape((-1, 1))
            summary(y, 'y')
            x_y = np.concatenate([x, y], axis=1)
            summary(x_y, 'energy_surface', highlight=True)
        else:
            raise NotImplementedError
        x, y = totsr(x), totsr(y)

    banner('feat')
    summary(x, 'x')
    summary(y, 'y')
    summary(labels_list, 'labels_list', exit=exit)
    return x, y, labels_list


import argparse

parser = argparse.ArgumentParser(description='activation map viz')
parser.add_argument('--mat', type=str, default='4_non_equal', help='dataset')
parser.add_argument('--log', action='store_true', help='log10 y')  # todo
parser.add_argument('--bm', action='store_true', help='use data from BM.csv')
parser.add_argument('--norm_w', action='store_true', help='normalize weight')
parser.add_argument('--emb', type=str, default='one_hot', help='', choices=['one_hot', 'neural', 'naive'])
parser.add_argument('--method', type=str, default='deepset')
parser.add_argument('--directory', type=str, default='Processed', help='')
parser.add_argument('--prop', type=str, default='bulk')
parser.add_argument('--idx', type=int, default=1, help='idx for deepset')

if __name__ == '__main__':
    banner(timestamp())
    args = parser.parse_args()
    x, y, labels_list = load_data(args, exit=False)
    x_screen, _, labels_list_screen = load_data(args, exit=False, screen=True)

    metrics = []
    for i in range(10):

        classifer = clf(x, y, label=labels_list, split=[0.6, 0.2, 0.2], norm_x=True, norm_y=True, regression=True)
        classifer.train_val_test(rs=i)
        for m in [args.method]:

            if args.method == 'deepset':
                clf_args, opt_param = load_deepset_yaml(id=args.idx)
                if args.mat == '5_non_equal':
                    clf_args['extractor_hidden_dim'] = [21]
                    clf_args['in_features'] = 21
                    clf_args['set_features'] = 21

            # eval
            if args.method in ['deepset']:
                getattr(classifer, m)(**clf_args, kw_opt=opt_param)
                metric = classifer.eval(verbose=True, tsr=True, viz=False)

                if args.mat != '5_non_equal':
                    y_screen = classifer.eval(tsr=True, x_test=x_screen)

            else:
                getattr(classifer, m)()
                metric = classifer.eval(verbose=True)
                if args.mat != '5_non_equal':
                    y_screen = classifer.eval(x_test=x_screen)

            metrics.append(metric)
            continue

    summary(np.array(metrics), name='final metric')
