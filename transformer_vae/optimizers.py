import torch
from transformers import Adafactor


class FixedAdafactor(Adafactor):
    '''
        Subclassed to use original `_approx_sq_grad` code.
        I found the old method broke when given a greator than 2-dim `exp_avg_sq_row`.
    '''

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor.unsqueeze(-1), c_factor)
