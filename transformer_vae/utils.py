from typing import List
import torch
import numpy as np
from torch.utils.data import Sampler


def assertEqual(actual, expected, msg, first="Got", second="Expected"):
    if actual != expected:
        raise ValueError(msg + f' {first}: "{actual}" {second}: "{expected}"')


def assertIn(actual, expected, msg, first="Got", second="Expected one of"):
    if actual not in expected:
        raise ValueError(msg + f' {first}: "{actual}" {second}: {expected}')


def slerp(ratio: float, t1: torch.FloatTensor, t2: torch.FloatTensor):
    '''
        Perform a spherical interpolation between 2 vectors.
        Most of the volume of a high-dimensional orange is in the skin, not the pulp.
        This also applies for multivariate Gaussian distributions.
        To that end we can interpolate between samples by following the surface of a n-dimensional sphere rather than a straight line.

        Args:
            ratio: Interpolation ratio.
            t1: Tensor1
            t2: Tensor2
    '''
    low_norm = t1 / torch.norm(t1, dim=1, keepdim=True)
    high_norm = t2 / torch.norm(t2, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - ratio) * omega) / so).unsqueeze(1) * t1 + (torch.sin(ratio * omega) / so).unsqueeze(1) * t2
    return res


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def fake_object(device, *attrs):
    '''
        Fake a Huggingface dataclass with desired attributes.
    '''
    empty_tensor = torch.tensor(0, dtype=torch.float, device=device)
    return AttrDict({atr: empty_tensor for atr in attrs})


class SortishSampler(Sampler):
    """
    Go through the text data by order of src length with a bit of randomness. From fastai repo.
    Modified to use shortest sequences first.
    """

    def __init__(self, data, shuffle=True):
        self.data, self.shuffle = data, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, 1, shuffle=self.shuffle)[::-1].tolist())


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx
