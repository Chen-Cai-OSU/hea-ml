# Created at 2020-08-10
# Summary:
# Created at 2020-07-20
# Summary: generate a collection of random 4-element heas and apply deepset

import random

import numpy as np

from code.utils.format import pf

seed = 42
np.random.seed(seed)
random.seed(42)
# D = data_util(log=False, bm = True, )
elements = ['Al', 'Cr', 'Nb', 'W', 'Mn', 'V', 'Fe', 'Co', 'Ti', 'Mo', 'Ni', 'Cu', 'Zr', 'Hf']  # D.metals
ratios = [.6, .7, .8, .9, 1]


def random_hea(elements, ratios):
    elements = np.random.choice(elements, 4, replace=False)
    elements.sort()
    ratios = np.random.choice(ratios, 4, replace=True)
    ratios[0] = 1
    hea = ''
    for i in range(4):
        hea += f'{elements[i]}_{ratios[i]}_'
    return hea[:-1]


def random_heas(n=100):
    np.random.seed(42)
    heas = []
    for _ in range(n):
        hea = random_hea(elements, ratios)
        heas.append(hea)
    heas = list(set(heas))
    heas.sort()
    return heas  # AlxCrCoNiFe() + heas


def AlxCrCoNiFe():
    heas = []
    for x in np.linspace(0.1, 0.5, num=5):
        x = pf(x, 2)
        heas.append(f'Al_{x}_Cr_1_Co_1_Ni_1_Fe_1')
    return heas
