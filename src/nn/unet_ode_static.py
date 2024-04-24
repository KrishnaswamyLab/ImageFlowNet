import os
import sys
import numpy as np
import torch
from .base import BaseNetwork
from .nn_utils import StaticODEfunc, ODEBlock

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/I2SB/')
from guided_diffusion.script_util import create_model
from guided_diffusion.unet import timestep_embedding


class StaticODEUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device,
                 in_channels: int,
                 ode_location: str = 'all_connections',
                 **kwargs):
        '''
        A UNet model with ODE.
        NOTE: This is a UNet with a static ODE vector field.

        Parameters
        ----------
        device: torch.device
        in_channels: int
            Number of input image channels.

        ode_location: str
            If 'bottleneck', only perform ODE on the bottleneck layer.
            If 'all_resolutions', skip connections with the same resolution share the same ODE.
            If 'all_connections', perform ODE separately in all skip connections.
        All other kwargs will be ignored.
        '''
        super().__init__()

        self.device = device
        self.ode_location = ode_location
        assert self.ode_location in ['bottleneck', 'all_resolutions', 'all_connections']
        image_size = 256  # TODO: currently hard coded

        # NOTE: This model is smaller than the other counterparts,
        # because running NeuralODE require some significant GPU space.
        # initialize model
        self.unet = create_model(
            image_size=image_size,
            in_channels=in_channels,
            num_channels=128,
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
        self.dim_list = []
        h_dummy = torch.zeros((1, 1, image_size, image_size)).type(self.unet.dtype)
        t_dummy = torch.zeros((1)).type(self.unet.dtype)
        emb = self.unet.time_embed(timestep_embedding(t_dummy, self.unet.model_channels))
        for module in self.unet.input_blocks:
            h_dummy = module(h_dummy, emb)
            self.dim_list.append(h_dummy.shape[1])
        h_dummy_bottleneck = self.unet.middle_block(h_dummy, emb)
        self.dim_list.append(h_dummy_bottleneck.shape[1])

        # Construct the ODE modules.
        self.ode_list = torch.nn.ModuleList([])
        if self.ode_location == 'bottleneck':
            self.ode_list.append(ODEBlock(StaticODEfunc(dim=h_dummy_bottleneck.shape[1])))
        elif self.ode_location == 'all_resolutions':
            for dim in np.unique(self.dim_list):
                self.ode_list.append(ODEBlock(StaticODEfunc(dim=dim)))
        elif self.ode_location == 'all_connections':
            for dim in self.dim_list:
                self.ode_list.append(ODEBlock(StaticODEfunc(dim=dim)))

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_grad: bool = False):
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

        # Provide a dummy time embedding, since we are learning a static ODE vector field.
        dummy_t = torch.zeros_like(t).to(t.device)
        emb = self.unet.time_embed(timestep_embedding(dummy_t, self.unet.model_channels))

        h = x.type(self.unet.dtype)

        # Contraction path.
        h_skip_connection = []
        for module in self.unet.input_blocks:
            h = module(h, emb)
            h_skip_connection.append(h)

        # Bottleneck
        h = self.unet.middle_block(h, emb)

        # ODE on bottleneck
        if use_ode:
            h = self.ode_list[-1](h, integration_time)

        # Expansion path.
        for module_idx, module in enumerate(self.unet.output_blocks):
            h_skip = h_skip_connection.pop(-1)

            # ODE over skip connections.
            if use_ode and self.ode_location in ['all_resolutions', 'all_connections']:
                if self.ode_location == 'all_connections':
                    curr_ode_block = self.ode_list[::-1][module_idx + 1]
                else:
                    resolution_idx = np.argwhere(np.unique(self.dim_list) == h_skip.shape[1]).item()
                    curr_ode_block = self.ode_list[resolution_idx]
                h_skip = curr_ode_block(h_skip, integration_time)

            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb)

        # Output.
        h = h.type(x.dtype)
        output = self.unet.out(h)

        if return_grad:
            vec_field_gradients = 0
            for i in range(len(self.ode_list)):
                vec_field_gradients += self.ode_list[i].vec_grad()
            return output, vec_field_gradients.mean() / len(self.ode_list)
        else:
            return output
