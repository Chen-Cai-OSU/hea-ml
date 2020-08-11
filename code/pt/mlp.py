# Created at 2020-08-10
# Summary:
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
from torch.nn import Sequential as Seq, ReLU, BatchNorm1d as BN
import warnings


class MLP_simple(nn.Module):

    def __init__(self, h_sizes, **kwargs):
        """
        https://bit.ly/3drTQ6C
        :param h_sizes: a list of num of hidden units
        :param out_size: output num of classes
        :param kwargs: just filter out useless kwargs that might be passed in accidentially
        """
        super(MLP_simple, self).__init__()

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            layer = Seq(nn.Linear(h_sizes[k], h_sizes[k + 1]), ReLU(), BN(h_sizes[k + 1]))
            self.hidden.append(layer)

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class MLP(nn.Module):

    def __init__(self, h_sizes, task='reg', input_dim=41, out_dim=1, nonlinear=nn.ReLU(),
                 **kwargs):
        """
        https://bit.ly/3drTQ6C
        :param h_sizes: a list of num of hidden units (exclude input/output dim)
        :param out_size: output num of classes
        :param kwargs: just filter out useless kwargs that might be passed in accidentially
        """
        # assert 'input_dim' in kwargs.keys()
        # assert 'out_dim' in kwargs.keys()
        # input_dim, out_dim = kwargs['input_dim'], kwargs['out_dim']

        super(MLP, self).__init__()
        self.task = task
        if h_sizes[0] == input_dim:
            warnings.warn(f'h_sizes[0] {h_sizes[0]} == input_dim {input_dim}')

        if h_sizes[-1] == out_dim:
            warnings.warn(f'h_sizes[-1] {h_sizes[-1]} == input_dim {out_dim} ')
        h_sizes = [input_dim] + h_sizes + [out_dim]
        # print(h_sizes)

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            lin = nn.Linear(h_sizes[k], h_sizes[k + 1])
            if k < len(h_sizes) - 2:
                if kwargs['bn']:
                    layer = Seq(lin, nonlinear, BN(h_sizes[k + 1]))
                else:
                    layer = Seq(lin, nonlinear)
            else:  # last layer
                layer = lin

            self.hidden.append(layer)

    def forward(self, x):
        # print(self.hidden)
        x = x.type(torch.float)
        for layer in self.hidden:
            x = layer(x)
            x = x.type(torch.float)

        if self.task == 'clf':
            x = F.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    x = torch.rand(100, 41)
    kwargs = {'bn': False}
    mlp = MLP([100], **kwargs)
    out = mlp(x)
    print(out)
