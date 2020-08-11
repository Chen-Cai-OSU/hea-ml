# Created at 2020-08-10
# Summary:
import os
import os.path as osp
from pprint import pprint

import torch
from os.path import expanduser
import platform


def detect_sys():
    sys = platform.system()

    if sys == 'Linux':
        return sys
    elif sys == 'Darwin':
        return sys
    else:
        NotImplementedError


def all_dirs(dir):
    """ list all directoreis under dir """
    return next(os.walk(dir))[1]  # http://bit.ly/39ZBOGq


def assert_file_exist(f):
    assert os.path.isfile(f), f'File {f} does not exist'


def log_dir():
    return f'{sig_dir()}../data/log'


def tb_dir():
    dir = f'{home()}/Documents/DeepLearning/Signor/data/tensorboard/'
    make_dir(dir)
    return dir


def make_dir(dir):
    # has side effect

    if dir == None:
        return

    if not os.path.exists(dir):
        os.makedirs(dir)


def home():
    # https://bit.ly/2ZPFE2R
    home = expanduser("~")
    return home


def hea_sys_dir():
    ret = os.path.dirname(__file__)
    ret = os.path.join(ret, '..', '..', 'data', 'HEA_System', '')
    return ret
    # return os.path.join(home(), 'Dropbox', 'Wei_Data', 'HEA_System', '')  # '/home/cai.507/Dropbox/Wei_Data/HEA_System/'


def pretrain_dir():
    return f'{home()}/Documents/DeepLearning/pretrain-gnns/'


def home_dir():
    from pathlib import Path
    home = str(Path.home())
    return home


def denali_dir():
    return os.path.join(sig_dir(), 'viz', 'denali', 'hea', '')


def hea_emb_dir():
    return os.path.join(sig_dir(), '..', 'data', 'hea_emb', '')


def sig_dir():
    if detect_sys() == 'Linux':
        return '/home/cai.507/Documents/DeepLearning/Signor/signor/'
    elif detect_sys() == 'Darwin':
        return '/Users/admin/Documents/osu/Research/Signor/signor/'
    else:
        NotImplementedError


def signor_dir():
    return f'{home()}/Documents/DeepLearning/Signor/signor/'


def get_dir(f):
    import os
    f = f'{home()}/Documents/DeepLearning/Signor/signor/graph/cgcnn/code/run_finetune.py'
    os.path.dirname(f)


def curdir():
    return osp.dirname(osp.realpath(__file__))


def find_files(dir='./', suffix='.txt', verbose=False, include_dir=False):
    # find all files in a dir ends with say .txt
    # https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
    assert dir[-1] == '/'
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            if include_dir: file = os.path.join(dir, file)
            if verbose: print(file)
            files.append(file)
    return files


def rm_files(dir='./', suffix='.txt', verbose=True):
    files = find_files(dir=dir, suffix=suffix, include_dir=True)
    rmfiles(files, verbose=False)
    if verbose:
        print(f'Remove {len(files)} files with suffix {suffix} at {dir}.')


def rmfiles(files, verbose=False):
    for f in files:
        assert os.path.isfile(f), f'{f} is not file'
        os.remove(f)
        if verbose: print(f'remove {f}')


def write_append(f):
    if os.path.exists(f):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    return append_write


def read_file(f):
    with open(f) as f:
        cont = f.readlines()
    return cont


if __name__ == '__main__':
    hea_sys_dir()
