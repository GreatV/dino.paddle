import torch.nn as nn
from functools import partial


class test(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func=nn.LayerNorm):
        super(test, self).__init__()
        self.norm = norm_func(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = test(10, 10, partial(nn.LayerNorm, eps=0.2))
