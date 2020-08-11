# Created at 2020-08-10
# Summary:
import torch
import numpy as np


def totsr(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    else:
        assert isinstance(arr, np.ndarray)
        return torch.Tensor(arr)


class Normalizer(object):
    """Normalize a Tensor and restore it later.
       Got from cgcnn
    """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        tensor = totsr(tensor)

        if isinstance(tensor, torch.Tensor):
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
        elif isinstance(tensor, np.ndarray):
            self.mean = np.mean(tensor)
            self.std = np.std(tensor)
        else:
            exit(f'Type {type(tensor)} not supported.')

    def norm(self, tensor):
        tensor = totsr(tensor)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        normed_tensor = totsr(normed_tensor)
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Multinormalizer():
    def __init__(self, tensor):
        self.tensor = totsr(tensor)
        self.mean, self.std = [], []
        assert self.tensor.ndimension() == 2
        n = self.tensor.size(1)
        for i in range(n):
            self.mean.append(torch.mean(self.tensor[:, i]))
            self.std.append(torch.std(self.tensor[:, i]))

    def norm(self, tensor):
        tensor = totsr(tensor)
        assert tensor.size(1) == self.tensor.size(1) == len(self.mean) == len(self.std)
        res = []
        for i in range(tensor.size(1)):
            t_i = (tensor[:, i] - self.mean[i]) / self.std[i]  # Normalizer(tensor[:, i]).norm(tensor[:, i])
            t_i = t_i.view(-1, 1)
            res.append(t_i)

        res = torch.cat(res, 1)
        assert res.size() == tensor.size()
        return res

    def denorm(self, tensor):
        tensor = totsr(tensor)
        assert tensor.size(1) == self.tensor.size(1) == len(self.mean) == len(self.std)
        res = []
        for i in range(tensor.size(1)):
            t_i = tensor[:, i] * self.std[i] + self.mean[i]  # (tensor[:, i] - self.mean[i]) / self.std[i]
            t_i = t_i.view(-1, 1)
            res.append(t_i)
        res = torch.cat(res, 1)
        assert res.size() == tensor.size()
        return res


if __name__ == '__main__':
    from signor.monitor.probe import summary

    x = [np.random.random((100, 1))]  # , 3*np.random.random((100, 1)), 10*np.random.random((100, 1))
    x = np.concatenate(x, axis=1)

    x1 = [np.random.random((100, 1)), 3 * np.random.random((100, 1)), 10 * np.random.random((100, 1))]
    x1 = np.concatenate(x1, axis=1)

    n = Multinormalizer(x)
    normx = n.norm(x)
    for i in range(x.shape[1]):
        summary(normx[:, i], name=f'normx[:, {i}]')

    print(n.mean)

    x_check = n.denorm(normx)
    summary(x_check, name='x_check')
    summary(totsr(x1), name='x1')
    summary(totsr(x), name='x')
