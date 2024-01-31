from .base import BaseNetwork
from .nn_utils import ConvBlock, ResConvBlock, timestep_embedding
from .common_encoder import Encoder
import os
import sys
import torch

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/I2SB/')
from guided_diffusion.script_util import create_model


class I2SBUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device,
                 in_channels: int,
                 step_to_t,
                 diffusion,
                 **kwargs):
        '''
        An UNet model for I2SB: Image-to-Image Schrodinger Bridge.

        Parameters
        ----------
        device: torch.device
        in_channels: int
            Number of input image channels.
        step_to_t: List
            A mapping from step index to time t.
        diffusion:
            A Diffusion object.
        All other kwargs will be ignored.
        '''
        super().__init__()

        self.device = device
        self.step_to_t = step_to_t
        self.diffusion = diffusion

        # initialize model
        self.model = create_model(
            image_size=256,  # TODO: currently hard coded
            in_channels=in_channels,
            num_channels=256,
            num_res_blocks=2,
            channel_mult='',
            learn_sigma=False,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions='32,16,8',
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False)

        self.model.eval()
        self.model.to(self.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        assert t.dim()==1 and t.shape[0] == x.shape[0]
        return self.model(x, t)

    @torch.no_grad()
    def ddpm_sampling(self, x_start, steps):
        '''
        Inference.
        '''

        x_start = x_start.to(self.device)

        def pred_x_end_fn(x, step):
            step = torch.full((x.shape[0],), step, device=self.device, dtype=torch.long)
            t = self.step_to_t[step]
            out = self.model(x, t)
            return self.compute_pred_x0(step, x, out, clip_denoise=False)

        xs, x_end_pred = self.diffusion.ddpm_sampling(
            steps, pred_x_end_fn, x_start, mask=None, ot_ode=False, log_steps=None, verbose=False,
        )

        return xs, x_end_pred

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of I2SB Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0
