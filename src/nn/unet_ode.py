import os
import sys
import torch
from .base import BaseNetwork
from .nn_utils import ODEfunc, ODEBlock

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/I2SB/')
from guided_diffusion.script_util import create_model
from guided_diffusion.unet import timestep_embedding


class ODEUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device,
                 in_channels: int,
                 **kwargs):
        '''
        A UNet model with ODE.

        Parameters
        ----------
        device: torch.device
        in_channels: int
            Number of input image channels.
        All other kwargs will be ignored.
        '''
        super().__init__()

        self.device = device
        image_size = 256  # TODO: currently hard coded

        # initialize model
        self.unet = create_model(
            image_size=image_size,
            in_channels=in_channels,
            num_channels=256,
            num_res_blocks=1,
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

        # Record the channel dimensions by passing in a dummy tensor.
        channel_dims = []
        h_dummy = torch.zeros((1, 1, image_size, image_size)).type(self.unet.dtype)
        t_dummy = torch.zeros((1)).type(self.unet.dtype)
        emb = self.unet.time_embed(timestep_embedding(t_dummy, self.unet.model_channels))
        for module in self.unet.input_blocks:
            h_dummy = module(h_dummy, emb)
            channel_dims.append(h_dummy.shape[1])
        h_dummy = self.unet.middle_block(h_dummy, emb)
        channel_dims.append(h_dummy.shape[1])

        # Construct the ODE modules.
        self.ode_list = torch.nn.ModuleList([])
        for dim in channel_dims:
            self.ode_list.append(ODEBlock(ODEfunc(dim=dim)))

        self.unet.to(self.device)
        self.ode_list.to(self.device)

    def time_independent_parameters(self):
        '''
        Parameters related to ODE.
        '''
        return set(self.parameters()) - set(self.ode_list.parameters())

    def freeze_time_independent(self):
        '''
        Freeze paramters that are time-independent.
        '''
        for p in self.time_independent_parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        h_skip_connection = []
        emb = self.unet.time_embed(timestep_embedding(t, self.unet.model_channels))

        assert len(self.unet.input_blocks) == len(self.ode_list) - 1

        h = x.type(self.unet.dtype)
        for i, module in enumerate(self.unet.input_blocks):
            h = module(h, emb)
            if use_ode:
                h = self.ode_list[i](h, integration_time)
            h_skip_connection.append(h)

        h = self.unet.middle_block(h, emb)
        if use_ode:
            h = self.ode_list[-1](h, integration_time)

        for module in self.unet.output_blocks:
            h = torch.cat([h, h_skip_connection.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return self.unet.out(h)

