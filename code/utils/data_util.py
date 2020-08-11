# Created at 2020-08-10
# Summary:

""" get all data """

import csv
import os
import re
import sys
import warnings
from pymatgen.core.structure import Structure

import numpy as np
import pandas as pd
from code.utils.atom_emb import atom_emb
from code.utils.list import sublist
from code.utils.probe import summary
from tqdm import tqdm

from code.utils.dir import all_dirs, assert_file_exist, sig_dir


class data_util():
    def __init__(self, log=True, bm = False, dir = '/home/cai.507/Dropbox/Wei_Data/HEA_System/Processed'):
        self.dir = dir
        self.heas = all_dirs(self.dir)  # ['CrNbCuFe', 'CuFeNiAl'...]
        self.heas = self.filter_heas(bm=bm)

        if bm:
            self.cols = ['composition', 'sim_method', 'cell', 'energy', 'bulk', 'equil_volume', 'lattice_constant', 'B']
        else:
            self.cols = ['composition', 'cell', 'energy', 'bulk', 'equil_volume', 'lattice_constant']

        if 'Processed_with_EC' in dir:
            self.cols = ['composition', 'sim_method', 'cell', 'energy', 'bulk', 'equil_volume', 'lattice_constant', 'B', 'c11', 'c12', 'c44']

        self.metals = ['Al', 'Cr', 'Nb', 'W', 'Mn', 'V', 'Fe', 'Co', 'Ti', 'Mo', 'Ni', 'Cu', 'Zr', 'Hf']
        self.atom_num = {'Cr': 24, 'Nb': 41, 'Mo': 42, 'V': 23, 'W': 74, 'Ti': 22, 'Cu': 29, 'Co': 27, 'Fe': 26,
                         'Ni': 28, 'Al': 12, 'Mn': 25, 'Hf': 72, 'Zr': 40, 'Ta':73}

        self.element_emb = atom_emb().load_()
        # self.set_neural_emb()
        self.naive_emb = np.eye(120)

        df = self.props(verbose=False, bm=bm)

        if log:
            df['bulk'] = np.log10(df['bulk'])
            df['energy'] = np.log10(abs(df['energy']))

        self.df = df

        # added to help denali viz
        # denali_hea = f'{signor_dir()}viz/denali/hea/'
        # cmd = f'rm {denali_hea}*{len(df)}.csv' # remove old csv files
        # os.system(cmd)
        # df.to_csv(f'{denali_hea}{timestamp()}-{len(df)}.csv')
        # self.heas = list(df['composition']) #remark: commented out for non-euqal heas
        # print(len(self.heas))
        # print(self.heas)


    ######### emb #########


    def set_neural_emb(self):
        raise NotImplementedError

        # """ set neural_emb following the similar format with self.element_emb
        # """
        # self.neural_emb = dict()
        #
        # for key in self.element_emb.keys():
        #     res = np.array(self.element_emb[str(key)])
        #     res = res.reshape((1, len(res)))
        #     v = neural_emb(x=res, verbose=False)
        #     self.neural_emb[key] = v
        # summary(self.neural_emb)

    ######### io #########

    def filter_heas(self, bm=False):
        heas = []

        for hea in self.heas:
            if bm:
                f = os.path.join(self.dir, hea, f'{hea}_BM.csv')
            else:
                f = os.path.join(self.dir, hea, f'{hea}.csv')
            if os.path.isfile(f):
                heas.append(hea)
            else:
                print(f'Exclude {hea} because {f} does not exist. (Likely being synced...)')

        self.bad_heas = [] #['CrCuAlHf','CrCoAlZr', 'CrVAlZr', 'CrVAlHf', 'CrCuAlZr', 'CoNiAlHf', 'CoAlMnHf', 'CrAlHfZr', 'CrNbAlZr', 'CrAlMnZr', 'CrCoAlHf', 'CoFeAlZr', 'AlMnHfZr', 'CrMoAlHf', 'CrNbAlHf', 'CrFeAlHf', 'CoNiAlZr', 'CrAlMnHf', 'CrNiAlHf', 'CrNiAlZr', 'CrMoAlZr', 'CrTiAlZr', 'CoAlMnZr', 'CoAlHfZr', 'CrFeAlZr', 'CrTiAlHf']
        heas = [hea for hea in heas if hea not in self.bad_heas]
        self.heas = heas
        return self.heas

    def read_prop(self, hea, bm=False):
        if bm:
            f = os.path.join(self.dir, hea, f'{hea}_BM.csv')
        else:
            f = os.path.join(self.dir, hea, f'{hea}.csv')
        assert_file_exist(f)
        with open(f, 'r') as f:
            data = list(csv.reader(f))
            # print(data)  # [['CrNbCuFe', 'bcc', '-3899.5618225920825', '138.3862892045082', '2.846365460031061']]
            # exit()
            # for bm: [['CrNbCuFe', 'birchmurnaghan', 'bcc', '-3899.5618225920825', '138.3862892045082', '2.846365460031061', '3.0803742391182927', '6.297856056967505']]
            return data

    def props(self, verbose=False, bm=False):
        all_props = []
        for hea in self.heas:
            prop = self.read_prop(hea, bm=bm)
            all_props += prop  # all props is a list of lists

        df = pd.DataFrame(all_props, columns=self.cols)
        if verbose: summary(df)

        if df.shape[1] > 8:
            df = df.astype({'energy': 'float', 'bulk': 'float',
                        'equil_volume': 'float', 'lattice_constant': 'float',
                        'c11': 'float', 'c12':'float', 'c44':'float'})
        else:
            df = df.astype({'energy': 'float', 'bulk': 'float',
                            'equil_volume': 'float', 'lattice_constant': 'float'})

        if bm:
            df_bad = df[df.sim_method!='birchmurnaghan']
            if len(df_bad)!=0:
                warnings.warn('df bad error!')
                # print(df_bad)
                new_bad_heas = list(df_bad['composition'])
                assert set(new_bad_heas) <= set(self.bad_heas), f'Latest bad heas: {new_bad_heas}'
                exit()

            df = df[df.sim_method!='bcc']
        return df

    def get_metals(self):
        metals = []
        for hea in self.heas:
            metals += re.findall('[A-Z][^A-Z]*', hea)  # http://bit.ly/39OW1za. return a list of length 4 for each hea
        metals = list(set(metals))
        return metals

    @staticmethod
    def get_compositions(hea):
        """

        :param hea: can be like Al_0.2_Co_1_Cr_1_Cu_1_Fe_1 or AlCoCrCuFe
        :return:
        """

        if len(hea.split('_')) == 10:
            indices = [0,2,4,6,8]
            return sublist(hea.split('_'), indices)
        elif len(hea.split('_')) == 8:
            indices = [0,2,4,6]
            return sublist(hea.split('_'), indices)
        else:
            return re.findall('[A-Z][^A-Z]*', hea)

    def get_weights(self, hea):
        # todo: need to make sure weights and compositions are aligned
        if len(hea.split('_')) == 10:
            indices = [1, 3, 5, 7, 9]
            res =  sublist(hea.split('_'), indices)
            return list(map(np.float, res))
        elif len(hea.split('_')) == 8:
            indices = [1, 3, 5, 7]
            res =  sublist(hea.split('_'), indices)
            return list(map(np.float, res))
        else:
            return [1,1,1,1]

    def get_atom_num(self, ele):
        return self.atom_num[ele]
        # return element(ele).atomic_number

    def get_cell_feat(self, hea):
        # print(self.df[self.df['composition'] == hea]['cell'].to_numpy())
        res = self.df[self.df['composition'] == hea]['cell'].to_numpy()
        assert len(res) == 1
        if res[0] == 'bcc':
            return np.array([[0]])
        elif res[0] == 'fcc':
            return np.array([[1]])
        else:
            raise Exception(f'Only expect bcc/fcc, not {res}')

    def get_lattice_constant(self, hea):
        """ similar to get_cell_feat """
        res = self.df[self.df['composition'] == hea]['lattice_constant'].to_numpy()
        assert len(res) == 1
        return np.array([[float(res[0])]])

    ######### x/y #########

    def get_feat(self, atom, emb='one_hot'):
        """
        embedding look up
        :param atom: int
        :return: array of shape (1, n)
        """
        assert emb in ['one_hot', 'neural', 'naive']

        if atom == -1:
            if emb == 'one_hot':
                return np.zeros((1, len(self.element_emb['1'])))
            elif emb == 'neural':
                return np.zeros((1, 64))
            else:
                return np.zeros((1, 120))


        assert 1 <= atom and atom <= 100, f'Atom number {atom} not in [1, 100].'

        if emb == 'one_hot':
            res = self.element_emb[str(atom)]
            n = 92
        elif emb == 'neural':
            res = self.neural_emb[str(atom)]
            n = 64
        else:
            res = self.naive_emb[int(atom)]
            n = 120

        res = np.array(res)
        res = res.reshape((1, n))
        return res

    def y(self, hea, prop='bulk'):
        """
        get the prop from hea
        :param hea:
        :return:
        """
        df_ = self.df[self.df['composition'] == hea.replace('_', '')]
        assert len(df_) == 1
        return df_[prop].to_numpy()


class cif():
    """ simple baseline for xt's data
        quick and dirty.
        share similarity with class data_util
    """
    def __init__(self, dir):
        self.dir = dir  # os.path.join(signor_dir(), 'graph', 'cgcnn', 'data_', 'sample-regression', '')
        self.element_emb = atom_emb().load_()
        for k in self.element_emb.keys():
            self.element_emb[k] = np.array([self.element_emb[k]])
        self.neural_emb = data_util().neural_emb
        self.naive_emb =  data_util().naive_emb

    def get_feat(self, f='1000041.cif', emb='one_hot'):
        """ get a feat of a cif file as baseline """
        g = Structure.from_file(self.dir + f)
        assert emb in ['one_hot', 'neural', 'naive']

        if emb == 'one_hot':
            dim = 92
        elif emb == 'neural':
            dim = 64
        else:
            dim = 120
        feat = np.zeros((1, dim))

        for num in g.atomic_numbers:
            if emb == 'one_hot':
                feat += self.element_emb[str(num)]
            elif emb == 'neural':
                feat += self.neural_emb[str(num)]
            else:
                feat += self.naive_emb[num].reshape((1, dim))
        return feat

    def get_xy(self, emb='one_hot'):
        """ get x, y from xie tian's directory """
        df_y = pd.read_csv(self.dir + 'id_prop.csv', names=['mp', 'y'])
        x = []
        for i, row in tqdm(df_y.iterrows(), total=df_y.shape[0]): #df_y.iterrows():
            feat = self.get_feat(f=row['mp'] + '.cif', emb=emb)
            x.append(feat)

        x = np.concatenate(x, axis=0)
        from signor.utils.np import rm_zerocol
        x = rm_zerocol(x)
        y = df_y['y'].to_numpy()
        summary(y, name='prop')
        return x, y


import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--prop', type=str, default='band_gap', help='')  # todo
parser.add_argument('--bm', action='store_true', help='use data from BM.csv')


def get_compositions(hea):
    """
    :param hea: can be like Al_0.2_Co_1_Cr_1_Cu_1_Fe_1 or AlCoCrCuFe
    :return:
    """

    if len(hea.split('_')) == 10:
        indices = [0, 2, 4, 6, 8]
        return sublist(hea.split('_'), indices)
    elif len(hea.split('_')) == 8:
        indices = [0, 2, 4, 6]
        return sublist(hea.split('_'), indices)
    else:
        return re.findall('[A-Z][^A-Z]*', hea)

if __name__ == '__main__':
    # summary(data_util().read_prop('CrNbCuFe'))

    args = parser.parse_args()
    bm = args.bm
    D = data_util(bm=bm, dir = '/home/cai.507/Dropbox/Wei_Data/HEA_System/4_ele_fully_random_result')
    # D = data_util(bm=bm, dir='/home/cai.507/Dropbox/Wei_Data/HEA_System/Processed')
    df = D.props(verbose=False, bm=bm)
    print(df.head(n = 100))

    df = df.sort_values(by=['bulk'])
    print(df.head(20))
    summary(df)
    exit()

    df = df.astype({'energy': 'float', 'bulk': 'float', 'equil_volume': 'float', 'lattice_constant': 'float'})
    # df['bulk'] = np.log10(df['bulk'])
    # df['energy'] = np.log10(abs(df['energy']))

    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    # df.plot.scatter(x="equil_volume", y="cell", ax=ax[0, 0])
    # df.plot.scatter(x="lattice_constant", y="cell", ax=ax[0, 1])
    # df.plot.scatter(x="equil_volume", y="energy", ax=ax[1, 0])
    # df.plot.scatter(x="equil_volume", y="lattice_constant", ax=ax[1, 1])

    fig, ax2 = plt.subplots(1, 4, figsize=(8, 2))
    print(ax2)
    # plt.locator_params(nbins=3)

    fig.tight_layout(pad=0)
    kw = {'bins':20, 'kind':"hist", 'grid': False, 'sharex': False, 'sharey': True, 'legend':False}
    df[['bulk']].plot(**kw, ax=ax2[0])
    df[['energy']].plot(**kw, ax=ax2[1])
    df[['equil_volume']].plot(**kw, ax=ax2[2])
    df[['lattice_constant']].plot(**kw, ax=ax2[3])
    img_dir = os.path.join(sig_dir(), 'graph', 'hea', 'paper', 'figure', '')
    fname = os.path.join(img_dir, f'hist.pdf')
    plt.savefig(fname, bbox_inches='tight')
    # df[['bulk', 'energy', 'equil_volume', 'lattice_constant']].plot(bins=30, kind="hist",
    #                                                                 subplots=True, grid=False, ax=(2,2), sharex=False, sharey=False)
    plt.show()

    sys.exit()

    arg = parser.parse_args()
    print(arg)

    D = data_util()
    D.get_lattice_constant('CrNbCuFe')
    # summary(D.df, 'D.df')
    exit()

    D.set_neural_emb()
    summary(D.element_emb)
    # dir = os.path.join(signor_dir(), 'graph', 'cgcnn', 'data_', 'sample-regression', '')
    # /home/cai.507/Dropbox/2020_Spring/Network/proj/data/TianXie/cif/mp-ids-3402/elasticity.K_VRH
    dir = os.path.join(xt_cif_dir(), 'mp-ids-46744', arg.prop, '')
    x, y = cif(dir).get_xy(emb='one_hot')

    summary(x, 'x')
    summary(y, 'y')
    hist(y, show=False, title=arg.prop)
    # exit()

    # x, y = dim_reducer().sampler((x, y), s=10000)
    # a baseline
    classifer = clf(x, y, split=[0.8, 0.2], norm_x=True, norm_y=True, regression=True)
    metrics = []
    for i in range(1):
        classifer.train_val_test(rs=i)
        for m in ['mlp']: # ['rf', 'gbt']: # ['svm', 'linear_reg', 'mlp']:
            getattr(classifer, m)()
            classifer.eval()  #
    exit()

    element_emb = atom_emb().load_()

