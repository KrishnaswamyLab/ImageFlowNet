from .base import BaseNetwork
from .nn_utils import ConvBlock, ResConvBlock, ODEfunc, ODEBlock
from .common_encoder import Encoder
import torch


class ODEUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 5,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A UNet model with ODE.

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

        # This is for the decoder.
        self.ode_list = torch.nn.ModuleList([])
        self.up_list = torch.nn.ModuleList([])
        self.up_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.ode_list.append(ODEBlock(ODEfunc(dim=n_f * 2 ** d)))
            self.up_conn_list.append(torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1))
            self.up_list.append(upconv_block(n_f * 2 ** d))
        self.ode_list = self.ode_list[::-1]
        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.ode_bottleneck = ODEBlock(ODEfunc(dim=n_f * 2 ** self.depth))
        self.out_layer = torch.nn.Conv2d(n_f, out_channels, 1)


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        Time embedding through ODE.
        '''

        assert x.shape[0] == 1

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        x, residual_list = self.encoder(x)

        if use_ode:
            x = self.ode_bottleneck(x, integration_time)

        for d in range(self.depth):
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=2,
                                                mode='bilinear',
                                                align_corners=False)
            if use_ode:
                res = self.ode_list[d](residual_list.pop(-1), integration_time)
            else:
                res = residual_list.pop(-1)
            x = torch.cat([x, res], dim=1)
            x = self.non_linearity(self.up_conn_list[d](x))
            x = self.up_list[d](x)

        output = self.out_layer(x)

        return output
