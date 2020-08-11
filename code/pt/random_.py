# Created at 2020-08-10
# Summary:
import torch

def random_mask(size=None, ratio=0.5):
    mask = torch.FloatTensor(*size).uniform_() > ratio
    return mask

def random_binary(size=None, ratio = 0.5):
    mask = random_mask(size=size, ratio=ratio)
    return mask.int()

def fix_seed():
    torch.manual_seed(0)

if __name__ == '__main__':
    size = (10, 1)

    print(random_mask(size=size, ratio=0.5))
    print(random_binary(size=size, ratio=0.5))

