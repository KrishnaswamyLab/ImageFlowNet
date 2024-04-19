import math
import torch
from torchdiffeq import odeint
import torchcde


class ODEfunc(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm2d(dim)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm2d(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.conv1(t, out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.conv2(t, out)
        out = self.relu(out)
        return out

class StaticODEfunc(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm2d(dim)
        self.conv1 = torch.nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm2d(dim)
        self.conv2 = torch.nn.Conv2d(dim, dim, 3, 1, 1)
        self.nfe = 0

    def forward(self, t, x):
        # `t` is a dummy variable here.
        self.nfe += 1
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu(out)
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

    def vec_grad(self, x):
        # return self.odefunc(0, x).abs().mean()
        w1 = self.odefunc.conv1.weight
        w2 = self.odefunc.conv2.weight
        return (w1**2).sum() + (w2**2).sum()

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class CDEBlock(torch.nn.Module):

    def __init__(self, cdefunc, interpolation: str = "linear"):
        super().__init__()
        self.cdefunc = cdefunc
        self.interpolation = interpolation

    def forward(self, x, integration_time):
        x_and_t = torch.cat([integration_time[:-1].unsqueeze(0).unsqueeze(2),
                             x.reshape(x.shape[0], -1).unsqueeze(0)],
                            dim=2)
        if self.interpolation == 'cubic':
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_and_t)
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            coeffs = torchcde.linear_interpolation_coeffs(x_and_t)
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0]) #[:, 1:].reshape(1, 128, 256, 256)

        integration_time = integration_time.type_as(x)
        import pdb
        pdb.set_trace()
        out = torchcde.cdeint(X=X, z0=X0, func=self.cdefunc, t=integration_time)
        import pdb
        pdb.set_trace()
        return out[1]

    @property
    def nfe(self):
        return self.cdefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.cdefunc.nfe = value


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
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.ReLU(inplace=True))

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
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.InstanceNorm2d(num_filters),
            torch.nn.ReLU(inplace=True))

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