# Created at 2020-08-10
# Summary:
import json

import numpy as np
import os.path as osp
from code.utils.np import np2set, index_of_firstx
from code.utils.probe import summary


class atom_emb():
    def __init__(self):
        self.dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', '')
        self.cutpt_dict = {0: -1, 1: 18, 2: 25, 3: 35, 4: 45, 5: 57, 6: 67, 7: 77, 8: 81, 9: 91}
        self.n_feat = 9
        self.load_emb()
        self.sub_atom_emb()

    def load_(self):

        elem_embedding_file = f'{self.dir}atom_init.json'
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        return elem_embedding

    def load_emb(self):
        """ load atom emb array of shape (100, 92)"""
        elem_embedding_file = f'{self.dir}atom_init.json'
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)

        atom_emb = dict()
        for key, value in elem_embedding.items():
            atom_emb[key] = np.array(value, dtype=float)
        atom_emb_arr = np.array(list(atom_emb.values()))
        self.atom_emb_arr = atom_emb_arr
        # assert_rows_unique(self.atom_emb_arr)
        return self.atom_emb_arr

    def sub_atom_emb(self):
        """ from originaal atom_emb_array of shape (100, 92)
            get a subset (62) of atoms whose has 9 full features
            return sub_atom_emb of shape (62, 92)
        """
        cnts = np.sum(self.atom_emb_arr, axis=1)  # count how many non-zeros
        indices = [i for i in range(len(cnts)) if cnts[i] == self.n_feat]
        assert len(indices) == 69
        self.sub_atom_emb_arr = self.atom_emb_arr[indices, :]
        return self.sub_atom_emb_arr

    def cvt(self):
        """ convert atom_emb into an array of shape (100, 9)
            where 9 represents 9 feats
        """
        cats = []
        for i in range(1, self.n_feat + 1):
            col_indices = range(self.cutpt_dict[i - 1] + 1, self.cutpt_dict[i] + 1)
            tmp = self.atom_emb_arr[:, col_indices]
            sub_tmp = self.sub_atom_emb_arr[:, col_indices]
            assert np.sum(sub_tmp) == sub_tmp.shape[0]  # each row there is exactly one 1.
            assert np2set(np.sum(sub_tmp, axis=1)) <= set([1])

            category = index_of_firstx(tmp, x=1, not_found=-1) + 1  # array of shape (100, )
            summary(category, 'category')
            cats.append(category)

        sol = np.concatenate(cats, axis=1)
        summary(sol, 'sol')
        for i in range(9):
            summary(sol[:, i], f'sol[:,{i}]')

        # assert_rows_unique(sol) # AssertionError: unique rows (92, 92). data (100, 92).
        return sol



