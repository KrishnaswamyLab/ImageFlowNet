import math
import torch
from torchdiffeq import odeint


class ODEfunc(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.GroupNorm(min(32, dim), dim)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.GroupNorm(min(32, dim), dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = torch.nn.GroupNorm(min(32, dim), dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(torch.nn.Module):

    def __init__(self, odefunc, tolerance: float = 1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.tolerance = tolerance

    def forward(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        out = odeint(self.odefunc,
                     x,
                     integration_time,
                     rtol=self.tolerance,
                     atol=self.tolerance)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ConvBlock(torch.nn.Module):

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.BatchNorm2d(num_filters),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.BatchNorm2d(num_filters),
            torch.nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class ResConvBlock(torch.nn.Module):

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.BatchNorm2d(num_filters),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.BatchNorm2d(num_filters),
            torch.nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x) + x


class ConcatConv2d(torch.nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 ksize=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 transpose=False):
        super(ConcatConv2d, self).__init__()
        module = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self._layer = module(dim_in + 1,
                             dim_out,
                             kernel_size=ksize,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


def timestep_embedding(timesteps, dim, max_period=10000):
    '''
    Create sinusoidal timestep embeddings.
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    '''
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding