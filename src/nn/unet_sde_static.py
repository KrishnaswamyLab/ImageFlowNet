import os
import sys
import numpy as np
import torch
from .base import BaseNetwork
from .nn_utils import StaticODEfunc, ODEfunc, SDEFunc, SDEBlock

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/I2SB/')
from guided_diffusion.script_util import create_model
from guided_diffusion.unet import timestep_embedding


class StaticSDEUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device,
                 in_channels: int,
                 sde_location: str = 'all_connections',
                 contrastive: bool = False,
                 **kwargs):
        '''
        A UNet model with SDE.
        NOTE: This is a UNet with a static SDE vector field.
        NOTE: Will rename to "PPSDEUNet" which stands for position-parameterized SDE UNet.

        Parameters
        ----------
        device: torch.device
        in_channels: int
            Number of input image channels.

        sde_location: str
            If 'bottleneck', only perform SDE on the bottleneck layer.
            If 'all_resolutions', skip connections with the same resolution share the same SDE.
            If 'all_connections', perform SDE separately in all skip connections.

        contrastive: bool
            Whether or not to perform contrastive learning (SimSiam) on bottleneck layer.

        All other kwargs will be ignored.
        '''
        super().__init__()

        self.device = device
        self.sde_location = sde_location
        assert self.sde_location in ['bottleneck', 'all_resolutions', 'all_connections']
        self.contrastive = contrastive

        image_size = 256  # TODO: currently hard coded

        # NOTE: This model is smaller than the other counterparts,
        # because running NeuralSDE require some significant GPU space.
        # initialize model
        self.unet = create_model(
            image_size=image_size,
            in_channels=in_channels,
            # num_channels=32,                 # avoid OOM for GBM dataset
            num_channels=64,
            num_res_blocks=1,
            channel_mult='',
            learn_sigma=False,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions='32,16,8',
            # num_heads=2,                     # avoid OOM for GBM dataset
            num_heads=4,
            num_head_channels=16,
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

        # Construct the SDE modules.
        self.sde_list = torch.nn.ModuleList([])

        if self.sde_location == 'bottleneck':
            self.sde_list.append(SDEBlock(SDEFunc(sde_mu=StaticODEfunc(dim=h_dummy_bottleneck.shape[1]))))
        elif self.sde_location == 'all_resolutions':
            for dim in np.unique(self.dim_list):
                self.sde_list.append(SDEBlock(SDEFunc(sde_mu=StaticODEfunc(dim=dim))))
        elif self.sde_location == 'all_connections':
            for dim in self.dim_list:
                self.sde_list.append(SDEBlock(SDEFunc(sde_mu=StaticODEfunc(dim=dim))))
        # if self.sde_location == 'bottleneck':
        #     self.sde_list.append(SDEBlock(SDEFunc(sde_mu=StaticODEfunc(dim=h_dummy_bottleneck.shape[1]),
        #                                           sde_sigma=ODEfunc(dim=h_dummy_bottleneck.shape[1]))))
        # elif self.sde_location == 'all_resolutions':
        #     for dim in np.unique(self.dim_list):
        #         self.sde_list.append(SDEBlock(SDEFunc(sde_mu=StaticODEfunc(dim=dim),
        #                                               sde_sigma=ODEfunc(dim=dim))))
        # elif self.sde_location == 'all_connections':
        #     for dim in self.dim_list:
        #         self.sde_list.append(SDEBlock(SDEFunc(sde_mu=StaticODEfunc(dim=dim),
        #                                               sde_sigma=ODEfunc(dim=dim))))

        self.unet.to(self.device)
        self.sde_list.to(self.device)

        if self.contrastive:
            pred_dim = 256
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(h_dummy_bottleneck.shape[1] *
                                h_dummy_bottleneck.shape[2] *
                                h_dummy_bottleneck.shape[3], pred_dim)
            )
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(pred_dim, pred_dim, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(pred_dim, pred_dim),
            )
            self.projector.to(self.device)
            self.predictor.to(self.device)

    def init_params(self):
        super().init_params()
        for block in self.sde_list:
            block.init_params()
        return

    def time_independent_parameters(self):
        '''
        Parameters related to SDE.
        '''
        return set(self.parameters()) - set(self.sde_list.parameters())

    def freeze_time_independent(self):
        '''
        Freeze paramters that are time-independent.
        '''
        for p in self.time_independent_parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_grad: bool = False, checkpointing: bool = True):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # Skip SDE if no time difference.
        use_sde = t.item() != 0
        if use_sde:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        # Provide a dummy time embedding, since we are learning a static SDE vector field.
        dummy_t = torch.zeros_like(t).to(t.device)
        emb = self.unet.time_embed(timestep_embedding(dummy_t, self.unet.model_channels))

        h = x.type(self.unet.dtype)

        # Contraction path.
        h_skip_connection = []
        for module in self.unet.input_blocks:
            if checkpointing and h.requires_grad and any(param.requires_grad for param in module.parameters()):
                h = torch.utils.checkpoint.checkpoint(module, h, emb)
            else:
                h = module(h, emb)
            h_skip_connection.append(h)

        # Bottleneck
        if checkpointing and h.requires_grad and any(param.requires_grad for param in self.unet.middle_block.parameters()):
            h = torch.utils.checkpoint.checkpoint(self.unet.middle_block, h, emb)
        else:
            h = self.unet.middle_block(h, emb)

        # SDE on bottleneck
        if use_sde:
            h = self.sde_list[-1](h, integration_time)

        # Expansion path.
        for module_idx, module in enumerate(self.unet.output_blocks):
            h_skip = h_skip_connection.pop(-1)

            # SDE over skip connections.
            if use_sde and self.sde_location in ['all_resolutions', 'all_connections']:
                if self.sde_location == 'all_connections':
                    curr_sde_block = self.sde_list[::-1][module_idx + 1]
                else:
                    resolution_idx = np.argwhere(np.unique(self.dim_list) == h_skip.shape[1]).item()
                    curr_sde_block = self.sde_list[resolution_idx]

                if checkpointing and h_skip.requires_grad and any(param.requires_grad for param in curr_sde_block.parameters()):
                    h_skip = torch.utils.checkpoint.checkpoint(curr_sde_block, h_skip, integration_time)
                else:
                    h_skip = curr_sde_block(h_skip, integration_time)

            h = torch.cat([h, h_skip], dim=1)
            if checkpointing and h.requires_grad and any(param.requires_grad for param in module.parameters()):
                h = torch.utils.checkpoint.checkpoint(module, h, emb)
            else:
                h = module(h, emb)

        # Output.
        h = h.type(x.dtype)
        if checkpointing and h.requires_grad and any(param.requires_grad for param in self.unet.out.parameters()):
            output = torch.utils.checkpoint.checkpoint(self.unet.out, h)
        else:
            output = self.unet.out(h)

        if return_grad:
            vec_field_gradients = 0
            for i in range(len(self.sde_list)):
                vec_field_gradients += self.sde_list[i].vec_grad()
            return output, vec_field_gradients.mean() / len(self.sde_list)
        else:
            return output

    def simsiam_project(self, x: torch.Tensor):
        # Provide a dummy time embedding, since we are learning a static SDE vector field.
        dummy_t = torch.zeros(1).to(x.device)
        emb = self.unet.time_embed(timestep_embedding(dummy_t, self.unet.model_channels))

        h = x.type(self.unet.dtype)
        # Contraction path.
        for module in self.unet.input_blocks:
            h = module(h, emb)
        # Bottleneck
        h = self.unet.middle_block(h, emb)

        h = h.reshape(h.shape[0], -1)

        z = self.projector(h)
        return z

    def simsiam_predict(self, z: torch.Tensor):
        p = self.predictor(z)
        return p

    @torch.no_grad()
    def return_embeddings(self, x: torch.Tensor, t: torch.Tensor):
        """
        Store and return the embedding vectors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        """
        embeddings_before = []
        embeddings_after = []

        # Skip SDE if no time difference.
        use_sde = t.item() != 0
        if use_sde:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        # Provide a dummy time embedding, since we are learning a static SDE vector field.
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

        # SDE on bottleneck
        embeddings_before.append(h)
        if use_sde:
            h = self.sde_list[-1](h, integration_time)
            embeddings_after.append(h)

        # Expansion path.
        for module_idx, module in enumerate(self.unet.output_blocks):
            h_skip = h_skip_connection.pop(-1)

            # SDE over skip connections.
            embeddings_before.append(h_skip)
            if use_sde and self.sde_location in ['all_resolutions', 'all_connections']:
                if self.sde_location == 'all_connections':
                    curr_sde_block = self.sde_list[::-1][module_idx + 1]
                else:
                    resolution_idx = np.argwhere(np.unique(self.dim_list) == h_skip.shape[1]).item()
                    curr_sde_block = self.sde_list[resolution_idx]

                h_skip = curr_sde_block(h_skip, integration_time)
                embeddings_after.append(h_skip)

            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb)

        return embeddings_before, embeddings_after
