from __future__ import print_function, division

import os.path as osp
from functools import partial

import numpy as np
import torch
from code.pt.mlp import MLP
from code.utils.dict import update_dict
from code.utils.format import timestamp, pf, banner, red
from code.utils.probe import summary
from code.utils.normalizer import Normalizer
from code.utils.np import tonp, totsr
from code.pt.deepset import DeepSet
from code.pt.random_ import fix_seed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, SVR
from skorch import NeuralNetRegressor

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def train_val_test(x, ratio=[0.5, 0.25, 0.25], verbose=0, norm=False, rs=42):
    """
    split data x (np.array or dataframe or list) into train, val, test
    :param x:
    :param ratio:
    :param verbose:
    :return:
    """
    assert len(ratio) == 3, 'Len of ratio should be 3'
    assert sum(ratio) == 1, print('sum of ratio is not 1.')
    first_split_ratio = 1 - ratio[0]
    second_split_ratio = ratio[2] / sum(ratio[1:])
    if norm:
        x = normalize(x, axis=0)  # normalize each feature

    x_train, x_rest = train_test_split(x, test_size=first_split_ratio, random_state=rs)
    x_val, x_test = train_test_split(x_rest, test_size=second_split_ratio, random_state=rs)
    if verbose == 1: print(f'shape of train/val/test is {x_train.shape}/{x_val.shape}/{x_test.shape}')

    return x_train, x_val, x_test


class clf():
    def __init__(self, x, y, label=None, split=None, norm_x=False, norm_y=False, regression=False):
        fix_seed()
        assert isinstance(x, (np.ndarray, torch.Tensor))
        assert isinstance(y, (np.ndarray, torch.Tensor))
        if label!=None:
            assert isinstance(label, list)
            assert x.shape[0] == y.shape[0]==len(label), f'x is of shape {x.shape}. y is of shpae {y.shape}. Label is of len {len(label)}'
        else:
            assert x.shape[0] == y.shape[0], f'x is of shape {x.shape}. y is of shpae {y.shape}.'

        self.x = x
        self.label = label
        self.norm_x = norm_x
        if self.norm_x:
            if np.ndim(x) == 2:
                self.x = normalize(x, axis=0)  # normalize each feature
            else:
                warnings.warn(f'Input x ndim is {np.ndim(x)}. Not normalized.')
                self.x = x
        self.y = y
        self.clf_name = None

        # todo: assert y is categorical vector
        self.norm_y = norm_y
        self.regression = regression
        self.config_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'configs', '')

        if split == None:
            self.train, self.val, self.test = 0.8, 0, 0.2
            self.split = None
        else:
            assert isinstance(split, list)
            assert len(split) <= 3
            assert sum(split) == 1
            self.split = split
            self.train, self.test = split[0], split[-1]
            self.val = 0 if len(split) == 2 else split[1]

    def _sample(self):
        """ sample train data for Normalizer """
        pass

    def train_val_test(self, rs=42):

        self.rs = rs
        if self.val == 0:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test, random_state=rs)
            x_val, y_val = None, None
        else:
            x_train, x_val, x_test = train_val_test(self.x, ratio=self.split, rs=rs)
            y_train, y_val, y_test = train_val_test(self.y, ratio=self.split, rs=rs)
            if self.label is not None:
                label_train, label_val, label_test = train_val_test(self.label, ratio=self.split, rs=rs)
                self.label_train, self.label_val, self.label_test = label_train, label_val, label_test
                self._label = np.concatenate([self.label_train, self.label_val, self.label_test], axis=0)

        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self._x = np.concatenate([self.x_train, self.x_val, self.x_test], axis=0)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self._y = np.concatenate([self.y_train, self.y_val, self.y_test], axis=0)

        if self.norm_y:
            normalizer = Normalizer(self.y_train)
            self.normalizer = normalizer

    def _model_summary(self):
        banner(f'{self.clf_name} summary.')
        print(self.clf)

    ################## classifier ###################

    def lr(self):
        self.clf_name = 'lr'
        if self.regression:
            raise NotImplementedError
        else:
            self.clf = LogisticRegression(random_state=0)

    def mlp(self, **kwargs):
        self.clf_name = 'mlp'
        if self.regression:
            self.clf = MLPRegressor(**kwargs, random_state=42)
        else:
            self.clf = MLPClassifier(**kwargs, random_state=42)
        # best_param = self.seach_hyper()
        best_param =  {'max_iter': 300, 'hidden_layer_sizes': [100, 100, 50], 'batch_size': 32, 'activation': 'relu', 'random_state': 42}
        print(red(best_param))
        self.clf = MLPRegressor(**best_param) if self.regression else MLPClassifier(**best_param)

    def deepset(self, kw_opt={}, **model_kw):
        """
        optization params has benn set
        :param model_kw:
        :return:
        """
        self.clf_name ='deepset'
        assert np.ndim(self.x_train) == 3, 'Deepset expects np.array of ndim 3.'

        default_kw_opt = {'verbose': 0, 'device': 'cpu',
                          'batch_size': 16,
                          'lr': 0.001,
                          'max_epochs': 300,
                          'optimizer__weight_decay': 0.0001,
                          'criterion': torch.nn.MSELoss,
                          'iterator_train__shuffle': True,
                           # optimizer__momentum=0.9,
                          'optimizer':torch.optim.Adam,} # self.seach_hyper()

        kw_opt = update_dict(kw_opt, default_kw_opt)
        print(red(kw_opt))

        if self.regression:
            self.clf = NeuralNetRegressor(DeepSet(**model_kw), **kw_opt)
        else:
            raise NotImplementedError
        self._model_summary()

    def mlp_skorch(self, kw_opt={}, **model_kw):
        self.clf_name = 'mlp_skorch'
        _mlp = partial(MLP, **model_kw) # dummy hidden num

        # important: still not good enough.
        default_kw_opt = {'verbose': 0,
                          'device': 'cpu',
                          'batch_size': 32,
                          'lr': 0.001,
                          'max_epochs': 300,
                          'optimizer__weight_decay': 0.0001,
                          'criterion': torch.nn.MSELoss,
                          # 'module__h_sizes': [100, 50],
                          'iterator_train__shuffle': True,
                          # optimizer__momentum=0.9,
                          'optimizer': torch.optim.Adam, }  # self.seach_hyper()
        kw_opt = update_dict(kw_opt, default_kw_opt)
        print(red(kw_opt))

        # opt_param =  {'verbose': 0, 'device': 'cuda:1', 'batch_size': 32, 'module__h_sizes': [100, 50]} # self.seach_hyper()
        # h_sizes = opt_param['module__h_sizes']
        kw_opt = update_dict(kw_opt, default_kw_opt)

        if self.regression:
            self.clf = NeuralNetRegressor(MLP([100, 50], **model_kw), **kw_opt)
        else:
            raise NotImplementedError

        self._model_summary()

    def knn(self):
        self.clf_name = 'knn'
        if self.regression:
            self.clf = KNeighborsRegressor()
        else:
            self.clf = KNeighborsClassifier()
        best_param = self.seach_hyper()
        self.clf = KNeighborsRegressor(**best_param) if self.regression else KNeighborsClassifier(**best_param)

    def linear_reg(self):
        self.clf_name = 'LinearRegression'
        if self.regression:
            self.clf = LinearRegression()
        else:
            raise NotImplementedError

    def linear_svm(self):
        self.clf_name = 'linear_svm'
        if self.regression:
            self.clf = SVR(kernel='linear')
        else:
            self.clf = SVC(kernel='linear')

    def rf(self):
        self.clf_name = 'rf'
        if self.regression:
            self.clf = RandomForestRegressor(random_state=42)
        else:
            self.clf = RandomForestClassifier(random_state=42)

        best_param = self.seach_hyper()
        best_param['random_state'] = 42
        self.clf = RandomForestRegressor(**best_param) if self.regression else RandomForestClassifier(**best_param)

    def gbt(self):
        self.clf_name = 'gbt'
        if self.regression:
            self.clf = GradientBoostingRegressor()
        else:
            self.clf = GradientBoostingClassifier()
        best_param = self.seach_hyper()
        self.clf = GradientBoostingRegressor(**best_param) if self.regression else GradientBoostingClassifier(**best_param)

    def svm(self, C=1000, gamma=1):
        self.clf_name = 'svm'

        if self.regression:
            self.clf = SVR(kernel='rbf', C=C, gamma=gamma)
        else:
            self.clf = SVC(gamma='auto', C=C)

        assert self.norm_x == True, f'Strongly encourage to norm x, otherwise svm will be very slow.'
        best_param = self.seach_hyper()
        self.clf = SVR(**best_param) if self.regression else SVC(**best_param)


    ################## hyperparameter search ###################

    def seach_hyper(self, **kwargs):

        """
        :param kwargs for RandomizedSearchCV.
        by default {'param_distributions':param_dist,'n_iter':20, 'cv':5, 'iid':False,
                    'n_jobs':-1, 'scoring':'neg_mean_squared_error', 'random_state':42}

        :return: best parameter
        """
        from signor.configs.scikit_hyper import hyperparameter_seach
        from signor.configs.util import load_configs

        hyper_dic = load_configs(self.config_dir + f'{self.clf_name}.yaml')

        x = np.concatenate((self.x_train, self.x_val), axis=0) if self.x_val is not None else self.x_train
        y = np.concatenate((self.y_train, self.y_val), axis=0) if self.y_val is not None else self.y_train

        if self.regression:
            y = tonp(self.normalizer.norm(y)) # important. added this one. will this change the mlp? no.
            reg_kwargs = {'scoring':'neg_mean_absolute_error'} #  neg_mean_squared_error
            # reg_kwargs = {'scoring': 'neg_mean_squared_error'}  #
            kwargs = reg_kwargs # update_dict(reg_kwargs, kwargs)
        else:
            clf_kwargs = {'scoring':'accuracy'}
            kwargs = clf_kwargs # update_dict(clf_kwargs, kwargs)
        # kwargs['n_iter'] = 100
        best_param = hyperparameter_seach(self.clf, x, y, param_dist=hyper_dic, model=self.clf_name, **kwargs)
        return best_param

    def set_param(self):
        best_param = self.seach_hyper()
        pass

    def save_model(self, net):
        """ save model. used for only skorch model """
        net.save_params(f_params='model.pkl', f_optimizer='opt.pkl', f_history='history.json')
        banner('Finish saving net.')

    ################## evaluation ###################


    def eval(self, verbose=True, tsr=False, **kwargs):
        """
        :param verbose:
        :param tsr: convert np.array to tensor
        ---------------
        :param all_data: evalualte on train+vali+test data
        :param x_test: replace replacing self.x_test with other data.
        :return:
        """
        if 'x_test' in kwargs: # replacing self.x_test with other data.
            summary(self.x_test, 'original x_test')
            summary(kwargs['x_test'], 'new x_test')
            self.x_test = kwargs['x_test']

        if 'all_data' in kwargs:
            self.x_test = self.x
            self.y_test = self.y

        if self.norm_y:
            assert self.regression == True

            if not tsr:
                target_normed = self.normalizer.norm(self.y_train)
                target_normed = tonp(target_normed).astype(self.y_train.dtype)
                target_normed = target_normed.reshape(self.y_train.shape)
                self.clf.fit(self.x_train, target_normed)
                y_pred_test = self.clf.predict(self.x_test)
            else:
                target_normed = self.normalizer.norm(self.y_train) # torch.tensor
                target_normed = target_normed.view(self.y_train.size())
                self.clf.fit(totsr(self.x_train), totsr(target_normed))
                y_pred_test = self.clf.predict(totsr(self.x_test))
                y_pred_test = tonp(y_pred_test)

            y_pred_test = tonp(self.normalizer.denorm(y_pred_test))

            if 'x_test' in kwargs:
                return y_pred_test

        else: # todo: handle torch tensor
            self.fit(self.x_train, self.y_train)
            y_pred_test = self.clf.predict(self.x_test)

        if 'all_data' in kwargs: self.y_pred_test = y_pred_test

        if self.regression:
            self.y_test = tonp(self.y_test)

            summary(y_pred_test, 'y_pred_test', highlight=True)
            summary(self.y_test, 'self.y_test', highlight=True)
            summary(abs(y_pred_test - self.y_test), 'diff', highlight=True)


            metric = np.sum(abs(y_pred_test - self.y_test)) / self.y_test.shape[0]
            relative_MAE = np.multiply(abs(y_pred_test - self.y_test), 1.0 /self.y_test)
            relative_MAE = np.mean(relative_MAE)
            print(f'relative mae (percent): {pf(relative_MAE * 100, 3)}')
            metric_name = 'MAE'
        else:
            metric = np.sum(y_pred_test == self.y_test) / self.y_test.shape[0]
            metric_name = 'Accuracy'

        if kwargs.get('viz', False) == True:
            from signor.viz.ml.regression import viz_reg_error
            viz_reg_error(y_pred_test, self.y_test)

        if verbose:
            banner(f'{self.clf_name}: {timestamp()}')
            print(f'norm_x: {self.norm_x}. norm_y: {self.norm_y}, seed: {self.rs} '
                  f'\n Training score ({self.x_train.shape[0]}): {pf(self.clf.score(self.x_train, self.y_train), 2)}.\
                    \n Test {metric_name} ({self.x_test.shape[0]}): {red(pf(metric, 3))}')

        return metric

    def pred_all(self):
        """ apply the trained model to all data. Used for plot confusion matrix.
            Call it after calling eval() method.
        """
        pass




if __name__ == '__main__':

    # deepset
    x, y = np.random.random((1000, 20, 30)), np.random.random((1000, 1))
    x = x.astype(np.double)  # need this.
    y = y.astype(np.double)
    x, y = totsr(x), totsr(y)

    for method in ['deepset']:  # ['knn', 'svm', 'rf', 'gbt']:
        classifer = clf(x, y, split=[0.6, 0.2, 0.2], norm_x=False, norm_y=True, regression=True)
        classifer.train_val_test()
        getattr(classifer, method)()
        classifer.eval(tsr=True)
    exit()

    # test regression
    x, y = np.random.random((1000, 30)), np.random.random((1000, 1))
    # train, val, test = load_mnist(data_dir='/home/cai.507/Documents/DeepLearning/Signor/data/')
    # x, y = sampler(train, s=1000, verbose=True)
    x = x.astype(np.double)  # need this.
    y = y.astype(np.double)
    x, y = totsr(x), totsr(y)

    for method in ['mlp_skorch']: # ['knn', 'svm', 'rf', 'gbt']:
        classifer = clf(x, y, split=[0.8, 0.2], norm_x=True, norm_y=True, regression=True)
        classifer.train_val_test()

        clf_args = {'task':'reg', 'input_dim': 30, 'out_dim': 1}
        getattr(classifer, method)(**clf_args)
        classifer.eval(tsr=True)

    exit()

