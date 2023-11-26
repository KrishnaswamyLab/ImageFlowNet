import torch

from .base import BaseNetwork
from .nn_utils import ConvBlock, ResConvBlock
from .common_encoder import Encoder


class AuxNet(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 5,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 dim_proj: int = 256,
                 non_linearity: str = 'relu'):
        '''
        Auxiliary Network that performs projection and segmentation.
        The projection head brings the model to a vector space of `dim_proj` dimensions,
        and performs contrastive learning. It then serves as a "discriminator" to train
        the main network through loss backpropagation.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        depth: int
            Depth of the model (encoding or decoding)
        use_residual: bool
            Whether to use residual connection within the same conv block
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        non_linearity : string
            One of 'relu' and 'softplus'
        '''
        super().__init__()

        self.device = device
        self.depth = depth
        self.use_residual = use_residual
        self.in_channels = in_channels
        self.dim_proj = dim_proj
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = torch.nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = torch.nn.Softplus()

        n_f = num_filters  # shorthand

        if self.use_residual:
            conv_block = ResConvBlock
            upconv_block = ResConvBlock
        else:
            conv_block = ConvBlock
            upconv_block = ConvBlock

        # This is for the encoder.
        self.encoder = Encoder(in_channels=in_channels,
                               n_f=n_f,
                               depth=self.depth,
                               conv_block=conv_block,
                               non_linearity=self.non_linearity)

        # This is for the segmentation head.
        self.up_list = torch.nn.ModuleList([])
        self.up_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.up_conn_list.append(torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1))
            self.up_list.append(upconv_block(n_f * 2 ** d))
        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.seg_head = torch.nn.ModuleList([
            conv_block(n_f),
            torch.nn.Conv2d(n_f, out_channels, 1),
            torch.nn.Sigmoid(),
        ])

        # This is for the projection head
        self.proj_head = torch.nn.ModuleList([
            conv_block(n_f * 2 ** self.depth),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(n_f * 2 ** self.depth, n_f * 2 ** self.depth, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(n_f * 2 ** self.depth, self.dim_proj, bias=False),
        ])

    def forward_seg(self, x: torch.Tensor):
        '''
        Forward through the segmentation path.
        '''

        x, residual_list = self.encoder(x)

        for d in range(self.depth):
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=2,
                                                mode='bilinear',
                                                align_corners=False)
            x = torch.cat([x, residual_list.pop(-1)], dim=1)
            x = self.non_linearity(self.up_conn_list[d](x))
            x = self.up_list[d](x)

        for module in self.seg_head:
            x = module(x)

        return x

    def forward_proj(self, x: torch.Tensor):
        '''
        Forward through the projection path.
        '''

        x, _ = self.encoder(x)

        for module in self.proj_head:
            x = module(x)

        # L2-normalize to put it onto the unit hypersphere.
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Please use `forward_seg` or `forward_proj` instead.')
