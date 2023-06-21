'''
Adapted from
https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models/blob/main/models/network.py
'''

import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from typing import List
from .unet import UNet


class I2IDiffusion(torch.nn.Module):
    '''
    Image2Image Diffusion Model.
    '''

    def __init__(
        self,
        in_channel: int = 3,
        inner_channel: int = 32,
        out_channel: int = 3,
        res_blocks: int = 2,
        attn_res: List = [16],
        num_head_channels: int = 32,
        dropout: float = 0.2,
        # beta_schedule={
        #     'train': {
        #         'schedule': 'linear',
        #         'n_timestep': 2000,
        #         'linear_start': 1e-6,
        #         'linear_end': 0.01
        #     },
        #     'test': {
        #         'schedule': 'linear',
        #         'n_timestep': 1000,
        #         'linear_start': 1e-4,
        #         'linear_end': 0.09
        #     }
        # }
    ):
        super().__init__()

        self.model = UNet(in_channel=in_channel,
                          inner_channel=inner_channel,
                          out_channel=out_channel,
                          res_blocks=res_blocks,
                          attn_res=attn_res,
                          num_head_channels=num_head_channels,
                          dropout=dropout)

        # self.beta_schedule = beta_schedule
        self.t_step = 0.1

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def forward(self, x_start, x_end, delta_t):
        b, *_ = x_start.shape

        # Sample a time for interpolation.
        t_arr = torch.linspace(0,
                               delta_t,
                               steps=int(delta_t / self.t_step) + 1)
        t = np.random.choice(t_arr, size=(b, ))
        t_interp = torch.from_numpy(t).to(x_start.device)

        # Interpolate a few points.
        t_ratio = t_interp / delta_t
        x_interp = (1 - t_ratio) * x_start + t_ratio * x_end

        x_interp_pred = self.model(x_start, t_interp)
        loss = self.loss_fn(x_interp, x_interp_pred)
        import pdb
        pdb.set_trace()
        return loss

    @torch.no_grad()
    def run(self, x_start, delta_t):
        x_end_pred = self.model(x_start, delta_t)
        return x_end_pred

    # def set_new_noise_schedule(self,
    #                            device=torch.device('cuda'),
    #                            phase='train'):
    #     to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
    #     betas = make_beta_schedule(**self.beta_schedule[phase])
    #     betas = betas.detach().cpu().numpy() if isinstance(
    #         betas, torch.Tensor) else betas
    #     alphas = 1. - betas

    #     timesteps, = betas.shape
    #     self.num_timesteps = int(timesteps)

    #     gammas = np.cumprod(alphas, axis=0)
    #     gammas_prev = np.append(1., gammas[:-1])

    #     # calculations for diffusion q(x_t | x_{t-1}) and others
    #     self.register_buffer('gammas', to_torch(gammas))
    #     self.register_buffer('sqrt_recip_gammas',
    #                          to_torch(np.sqrt(1. / gammas)))
    #     self.register_buffer('sqrt_recipm1_gammas',
    #                          to_torch(np.sqrt(1. / gammas - 1)))

    #     # calculations for posterior q(x_{t-1} | x_t, x_0)
    #     posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
    #     # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    #     self.register_buffer(
    #         'posterior_log_variance_clipped',
    #         to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
    #     self.register_buffer(
    #         'posterior_mean_coef1',
    #         to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
    #     self.register_buffer(
    #         'posterior_mean_coef2',
    #         to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    # def predict_start_from_noise(self, y_t, t, noise):
    #     return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
    #             extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise)

    # def q_posterior(self, y_0_hat, y_t, t):
    #     posterior_mean = (
    #         extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
    #         extract(self.posterior_mean_coef2, t, y_t.shape) * y_t)
    #     posterior_log_variance_clipped = extract(
    #         self.posterior_log_variance_clipped, t, y_t.shape)
    #     return posterior_mean, posterior_log_variance_clipped

    # def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
    #     noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
    #     y_0_hat = self.predict_start_from_noise(y_t,
    #                                             t=t,
    #                                             noise=self.denoise_fn(
    #                                                 torch.cat([y_cond, y_t],
    #                                                           dim=1),
    #                                                 noise_level))

    #     if clip_denoised:
    #         y_0_hat.clamp_(-1., 1.)

    #     model_mean, posterior_log_variance = self.q_posterior(y_0_hat=y_0_hat,
    #                                                           y_t=y_t,
    #                                                           t=t)
    #     return model_mean, posterior_log_variance

    # def q_sample(self, y_0, sample_gammas, noise=None):
    #     noise = default(noise, lambda: torch.randn_like(y_0))
    #     return (sample_gammas.sqrt() * y_0 +
    #             (1 - sample_gammas).sqrt() * noise)

    # @torch.no_grad()
    # def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
    #     model_mean, model_log_variance = self.p_mean_variance(
    #         y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
    #     noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
    #     return model_mean + noise * (0.5 * model_log_variance).exp()

    # @torch.no_grad()
    # def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
    #     b, *_ = y_cond.shape

    #     assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
    #     sample_inter = (self.num_timesteps // sample_num)

    #     y_t = default(y_t, lambda: torch.randn_like(y_cond))
    #     ret_arr = y_t
    #     for i in tqdm(reversed(range(0, self.num_timesteps)),
    #                   desc='sampling loop time step',
    #                   total=self.num_timesteps):
    #         t = torch.full((b, ), i, device=y_cond.device, dtype=torch.long)
    #         y_t = self.p_sample(y_t, t, y_cond=y_cond)
    #         if mask is not None:
    #             y_t = y_0 * (1. - mask) + mask * y_t
    #         if i % sample_inter == 0:
    #             ret_arr = torch.cat([ret_arr, y_t], dim=0)
    #     return y_t, ret_arr

    # def forward(self, y_0, y_cond=None, noise=None):
    #     # sampling from p(gammas)
    #     b, *_ = y_0.shape
    #     t = torch.randint(1, self.num_timesteps, (b, ),
    #                       device=y_0.device).long()
    #     gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
    #     sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
    #     sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand(
    #         (b, 1), device=y_0.device) + gamma_t1
    #     sample_gammas = sample_gammas.view(b, -1)

    #     noise = default(noise, lambda: torch.randn_like(y_0))
    #     y_noisy = self.q_sample(y_0=y_0,
    #                             sample_gammas=sample_gammas.view(-1, 1, 1, 1),
    #                             noise=noise)

    #     noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1),
    #                                 sample_gammas)
    #     loss = self.loss_fn(noise, noise_hat)
    #     return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start,
                                      linear_end,
                                      warmup_time,
                                      dtype=np.float64)
    return betas


def make_beta_schedule(schedule,
                       n_timestep,
                       linear_start=1e-6,
                       linear_end=1e-2,
                       cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start**0.5,
                            linear_end**0.5,
                            n_timestep,
                            dtype=np.float64)**2
    elif schedule == 'linear':
        betas = np.linspace(linear_start,
                            linear_end,
                            n_timestep,
                            dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == 'cosine':
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep +
            cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
