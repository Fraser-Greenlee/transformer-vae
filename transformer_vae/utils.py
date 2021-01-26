import torch


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
