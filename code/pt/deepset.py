# Created at 2020-08-10
# Summary:

# Created at 2020-04-07
# Summary: implement a generic deepset. modified from https://github.com/manzilzaheer/DeepSets/blob/master/PopStats/model.py
from functools import partial

import torch
import torch.nn as nn

from code.pt.random_ import fix_seed

fix_seed()
kwargs = {'extractor_hidden_dim': [50, 100],
          'regressor_hidden_dim': [100, 100, 50],
          'extractor_nl': nn.ELU,
          'regressor_nl': nn.ELU
          }


class DeepSet(nn.Module):

    def __init__(self, in_features=94, set_features=50, bn=False, **kwargs):
        """

        :param in_features:
        :param set_features:
        :param bn:
        :param weight: weights for each element in set
        :param kwargs:
        """
        super(DeepSet, self).__init__()

        print(f'DeepSet kw: {kwargs}')
        extractor_hidden_dim = kwargs['extractor_hidden_dim']
        regressor_hidden_dim = kwargs['regressor_hidden_dim']
        extractor_nl = kwargs['extractor_nl']
        regressor_nl = kwargs['regressor_nl']

        self.in_features = in_features
        self.out_features = set_features
        self.bn = bn

        self.feature_extractor = self.mlp(in_features, set_features, extractor_hidden_dim, nl=extractor_nl, bn=False,
                                          last_nn=False)

        self.agg = partial(torch.sum, dim=1)  # torch.sum(dim=1)
        self.regressor = self.mlp(set_features, 1, regressor_hidden_dim, nl=regressor_nl, bn=self.bn, bn_dim=1)

        self.add_module('feature_extractor', self.feature_extractor)
        self.add_module('regressor', self.regressor)

    def mlp(self, input_dim, output_dim, hidden, nl=nn.ELU, bn=False, bn_dim=1, last_nn=False):
        assert isinstance(hidden, list)

        hidden = [input_dim] + hidden + [output_dim]
        n = len(hidden)
        modules = []
        for i in range(n - 2):
            inner_module = []
            inner_module.append(nn.Linear(hidden[i], hidden[i + 1]))
            inner_module.append(nl(inplace=True))
            if bn:
                if bn_dim == 1:
                    inner_module.append(nn.BatchNorm1d(hidden[i + 1]))
                elif bn_dim == 2:
                    inner_module.append(nn.BatchNorm2d(hidden[i + 1]))
                elif bn_dim == 3:
                    inner_module.append(nn.BatchNorm3d(hidden[i + 1]))
                else:
                    raise NotImplementedError
            modules.append(nn.Sequential(*inner_module))

        modules.append(nn.Linear(hidden[n - 2], hidden[n - 1]))
        if last_nn:
            modules.append(nn.ELU(inplace=True))
        return nn.Sequential(*modules)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()

    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = self.agg(x)  #
        x = self.regressor(x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'Feature Exctractor=' + str(self.feature_extractor) \
               + '\n Set Feature' + str(self.regressor) + ')'


if __name__ == '__main__':
    # x = [torch.rand(20, 4, 30)] * 2
    x = torch.rand(20, 4, 30)
    model = DeepSet(in_features=30, set_features=10, **kwargs)

    ds = DeepSet(in_features=30, set_features=10, **kwargs)
    out = ds(x)
    print(out)
