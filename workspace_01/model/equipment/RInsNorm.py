
import torch
import torch.nn as nn

class RInsNorm(nn.Module):
    """
    Reverse Instance Normalize
    """
    def __init__(self, configs):
        super(RInsNorm,self).__init__()
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

    def Activate_InsNorm(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x * self.affine_weight + self.affine_bias
        return x

    def Deactivate_InsNorm(self, x):
        x = x - self.affine_bias
        x = x / (self.affine_weight + 1e-10)
        stdev = stdev[:,:,-1:]
        means = means[:,:,-1:]
        x = x * stdev
        x = x + means
        return x 

