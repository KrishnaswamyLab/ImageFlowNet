import math
import numpy as np
import torch
from torchdiffeq import odeint
import torchcde
import torchsde

class ODEfunc(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm2d(dim)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm2d(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.dim = dim
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
        self.dim = dim
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
        return out[-1]  # equivalent to `out[1]` if len(integration_time) == 2.

    def flow_field_norm(self, x):
        return torch.norm(self.odefunc(x), p=2)

    def vec_grad(self):
        '''
        NOTE: Only taking care of Conv2d weights.
        '''
        sum_weight_sq_norm = 0
        for m in self.odefunc.modules():
            if isinstance(m, torch.nn.Conv2d):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class SDEFunc(torch.nn.Module):
    '''
    Stochastic Differential Equation Func.

    It has to include class 2 methods:
    self.f: the drift term.
    self.g: the diffusion term.

    NOTE: self.noise_type and self.sde_type are required for torchsde.
    '''
    # def __init__(self, sde_mu, sde_sigma, noise_type='general', sde_type='ito'):
    #     super().__init__()
    #     self.sde_mu = sde_mu  # drift term
    #     self.sde_sigma = sde_sigma  # diffusion term
    #     self.noise_type = noise_type
    #     self.sde_type = sde_type

    #     assert self.sde_mu.dim == self.sde_sigma.dim
    #     self.dim = self.sde_mu.dim
    def __init__(self, sde_mu, sde_sigma=0.5, noise_type='diagonal', sde_type='ito'):
        super().__init__()
        self.sde_mu = sde_mu  # drift term
        # self.sde_sigma = sde_sigma # diffusion term
        self.sde_sigma = torch.nn.Parameter(torch.tensor(sde_sigma), requires_grad=True) # diffusion term
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.dim = self.sde_mu.dim

    # calculates the drift
    def f(self, t, x):
        '''
        Assuming x is a flattened tensor of [B, C, H, W] and H == W.
        '''
        x_spatial_dim = int(np.sqrt(x.shape[-1] / self.dim))
        out = x.reshape(x.shape[0], self.dim, x_spatial_dim, x_spatial_dim)
        sde_drift = self.sde_mu(t, out)
        return sde_drift.reshape(sde_drift.shape[0], -1)

    # calculates the diffusion
    # def g(self, t, x):
    #     # Assuming a 1-dimensional Brownian motion.
    #     x_spatial_dim = int(np.sqrt(x.shape[-1] / self.dim))
    #     out = x.reshape(x.shape[0], self.dim, x_spatial_dim, x_spatial_dim)
    #     sde_diffusion = self.sde_sigma(t, out)
    #     return sde_diffusion.reshape(sde_diffusion.shape[0], -1, 1)

    def g(self, t, x):
        # Assuming a 1-dimensional Brownian motion.
        return self.sde_sigma.expand_as(x)

    def init_params(self):
        '''
        Initialization trick from Glow.
        '''
        pass
        # Don't have linear layer in it.
        # self.sde_func_drift[-1].weight.data.fill_(0.)
        # self.sde_func_drift[-1].bias.data.fill_(0.)
        # self.sde_func_diffusion[-1].weight.data.fill_(0.)
        # self.sde_func_diffusion[-1].bias.data.fill_(0.)

class SDEBlock(torch.nn.Module):
    '''
    Stochastic Differential Equation block.
    '''

    def __init__(self,
                 sdefunc,
                 tolerance: float = 1e-3,
                 adjoint: bool = False):

        super().__init__()

        self.sdefunc = sdefunc
        self.tolerance = tolerance
        self.adjoint = adjoint

    def forward(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        x = x.reshape(x.shape[0], -1)

        sde_int = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        out = sde_int(self.sdefunc,
                      x,
                      integration_time,
                      dt=5e-2, # 1e-3 is too slow.
                      method='euler', # otherwise OOM
                      rtol=self.tolerance,
                      atol=self.tolerance)
        out_spatial_dim = int(np.sqrt(out.shape[-1] / self.sdefunc.dim))
        out = out.reshape(out.shape[0], self.sdefunc.dim, out_spatial_dim, out_spatial_dim)
        return out[-1].unsqueeze(0)

    def init_params(self):
        self.sdefunc.init_params()

    def vec_grad(self):
        '''
        NOTE: Only taking care of Conv2d weights.
        '''
        sum_weight_sq_norm = 0
        for m in self.sdefunc.modules():
            if isinstance(m, torch.nn.Conv2d):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

    @torch.no_grad()
    def forward_traj(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        x = x.reshape(x.shape[0], -1)

        sde_int = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        out = sde_int(self.sdefunc,
                      x,
                      integration_time,
                      dt=1e-4,
                      method='euler', # otherwise OOM
                      rtol=self.tolerance,
                      atol=self.tolerance)
        out_spatial_dim = int(np.sqrt(out.shape[-1] / self.sdefunc.dim))
        out = out.reshape(out.shape[0], self.sdefunc.dim, out_spatial_dim, out_spatial_dim)
        return out


class LatentClassifier(torch.nn.Module):
    '''
    A simple classifier model that produces
    a scalar (-inf, inf) from a tensor.
    '''

    def __init__(self, dim, emb_channels):
        super().__init__()
        self.emb_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(emb_channels, dim),
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm2d(dim)
        self.conv1 = torch.nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm2d(dim)
        self.conv2 = torch.nn.Conv2d(dim, dim, 3, 1, 1)
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, z, emb):
        out = self.norm1(z)
        out = self.conv1(out)
        out = self.relu(out)

        emb_out = self.emb_layer(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]
        out = out + emb_out

        out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = out.mean([2, 3])  # global average pooling
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out

class Combine2Channels(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm2d(dim * 2)
        self.conv1 = torch.nn.Conv2d(dim * 2, dim, 3, 1, 1)

    def forward(self, z_concat):
        out = self.norm1(z_concat)
        out = self.conv1(out)
        out = self.relu(out)
        return out

class SODEBlock(torch.nn.Module):
    '''
    State-augmented ODE block.
    z_tj = z_ti + \int_ti^tj f_theta(z_tau, z_s) d tau
    z_s = \sum_k gamma_k z_tk, tk < tau
    gamma_k = softmax(g(z_tk)), tk < tau
    '''

    def __init__(self,
                 odefunc,
                 combine_2channels,
                 latent_cls,
                 tolerance: float = 1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.combine_2channels = combine_2channels
        self.latent_cls = latent_cls
        self.tolerance = tolerance

    def forward(self, z_arr, emb, integration_time):
        integration_time = integration_time.type_as(z_arr)
        # We only integrate from the second-last to the last time.
        if len(integration_time) == 1:
            integration_time = torch.tensor([0, integration_time[0]]).float().to(z_arr.device)
        else:
            integration_time = torch.tensor([0, integration_time[-1] - integration_time[-2]]).float().to(z_arr.device)

        num_obs = z_arr.shape[0]
        if num_obs == 1:
            z_cat = torch.cat([z_arr[0].unsqueeze(0),
                               z_arr[0].unsqueeze(0)], dim=1)
        else:
            cls_outputs = self.latent_cls(z=z_arr, emb=emb)
            coeffs = torch.nn.functional.softmax(cls_outputs, dim=0)
            assert len(coeffs.shape) == 2 and coeffs.shape[1] == 1
            coeffs = coeffs.unsqueeze(-1).unsqueeze(-1)
            zs = (coeffs * z_arr).sum(dim=0, keepdim=True)
            z_cat = torch.cat([z_arr[0].unsqueeze(0),
                               zs], dim=1)

        z = self.combine_2channels(z_cat)
        out = odeint(self.odefunc,
                     z,
                     integration_time,
                     rtol=self.tolerance,
                     atol=self.tolerance)

        return out[1]

    def vec_grad(self):
        '''
        NOTE: Only taking care of Conv2d weights.
        '''
        sum_weight_sq_norm = 0
        for m in self.odefunc.modules():
            if isinstance(m, torch.nn.Conv2d):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

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