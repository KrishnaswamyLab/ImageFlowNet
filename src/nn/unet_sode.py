import os
import sys
import numpy as np
import torch
from .base import BaseNetwork
from .nn_utils import StaticODEfunc, Combine2Channels, LatentClassifier, SODEBlock

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/I2SB/')
from guided_diffusion.script_util import create_model
from guided_diffusion.unet import timestep_embedding


class SODEUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device,
                 in_channels: int,
                 **kwargs):
        '''
        A UNet model with State-augmented Ordinary Differential Equation (SODE).

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
            if h_dummy.shape[1] not in self.dim_list:
                self.dim_list.append(h_dummy.shape[1])
        h_dummy = self.unet.middle_block(h_dummy, emb)
        if h_dummy.shape[1] not in self.dim_list:
            self.dim_list.append(h_dummy.shape[1])

        # Construct the SODE modules.
        self.sode_list = torch.nn.ModuleList([])
        for dim in self.dim_list:
            # NOTE: not a typo. ODEfunc inside CDEBlock.
            self.sode_list.append(SODEBlock(StaticODEfunc(dim=dim),
                                            Combine2Channels(dim=dim),
                                            LatentClassifier(dim=dim, emb_channels=emb.shape[1])))

        self.unet.to(self.device)
        self.sode_list.to(self.device)

    def time_independent_parameters(self):
        '''
        Parameters related to CDE.
        '''
        return set(self.parameters()) - set(self.sode_list.parameters())

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
        use_ode = not (len(t) == 1 and t.item() == 0)
        if use_ode:
            integration_time = t.float()

        h_skip_connection = []

        # Provide a dummy time embedding, since we are learning a static ODE vector field.
        dummy_t = torch.zeros_like(t).to(t.device)
        emb = self.unet.time_embed(timestep_embedding(dummy_t, self.unet.model_channels))

        # State-augmented ODE actually needs the proper time embedding.
        emb_sode = self.unet.time_embed(timestep_embedding(t, self.unet.model_channels))

        h = x.type(self.unet.dtype)
        for module in self.unet.input_blocks:
            h = module(h, emb)
            if use_ode:
                ode_idx = np.argwhere(np.array(self.dim_list) == h.shape[1]).item()
                h_skip = self.sode_list[ode_idx](h, emb_sode, integration_time)
                h_skip_connection.append(h_skip)
            else:
                h_skip_connection.append(h)

        h = self.unet.middle_block(h, emb)
        if use_ode:
            ode_idx = np.argwhere(np.array(self.dim_list) == h.shape[1]).item()
            h = self.sode_list[ode_idx](h, emb_sode, integration_time)

        for module in self.unet.output_blocks:
            # When modeling multiple observations, we only keep the first observation.
            if h.shape[0] > 1:
                h = h[0].unsqueeze(0)
            h = torch.cat([h, h_skip_connection.pop(-1)], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)

        if return_grad:
            vec_field_gradients = 0
            for i in range(len(self.sode_list)):
                vec_field_gradients += self.sode_list[i].vec_grad()
            return self.unet.out(h), vec_field_gradients.mean() / len(self.sode_list)
        else:
            return self.unet.out(h)

