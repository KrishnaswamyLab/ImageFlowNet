from .base import BaseNetwork
from .nn_utils import ConvBlock, ResConvBlock, StaticODEfunc, ODEBlock
import torch


class StaticODEUNetSimple(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 64,
                 depth: int = 6,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu',
                 use_bn: bool = True):
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
        self.use_bn = use_bn

        n_f = num_filters  # shorthand

        if self.use_residual:
            conv_block = ResConvBlock
            upconv_block = ResConvBlock
        else:
            conv_block = ConvBlock
            upconv_block = ConvBlock

        # This is for the contraction path.
        self.conv1x1 = torch.nn.Conv2d(in_channels, n_f, 1, 1)
        self.down_list = torch.nn.ModuleList([])
        self.down_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            if self.use_bn:
                self.down_conn_list.append(torch.nn.Sequential(
                    torch.nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1),
                    torch.nn.BatchNorm2d(n_f * 2 ** (d + 1)),
                ))
            else:
                self.down_conn_list.append(torch.nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))
        self.bottleneck = conv_block(n_f * 2 ** self.depth)

        # This is for the expansion path.
        self.ode_list = torch.nn.ModuleList([])
        self.up_list = torch.nn.ModuleList([])
        self.up_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.ode_list.append(ODEBlock(StaticODEfunc(dim=n_f * 2 ** d)))
            self.up_list.append(upconv_block(n_f * 2 ** d))
            if self.use_bn:
                self.up_conn_list.append(torch.nn.Sequential(
                    torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1),
                    torch.nn.BatchNorm2d(n_f * 2 ** d),
                ))
            else:
                self.up_conn_list.append(torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1))
        self.ode_list = self.ode_list[::-1]
        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.ode_bottleneck = ODEBlock(StaticODEfunc(dim=n_f * 2 ** self.depth))
        self.out_layer = torch.nn.Conv2d(n_f, out_channels, 1)

    def time_independent_parameters(self):
        '''
        Parameters related to ODE.
        '''
        return set(self.parameters()) - set(self.ode_list.parameters()) - set(self.ode_bottleneck.parameters())

    def freeze_time_independent(self):
        '''
        Freeze paramters that are time-independent.
        '''
        for p in self.time_independent_parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_grad: bool = False):
        '''
        Time embedding through ODE.
        '''

        assert x.shape[0] == 1

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        ######################
        # Contraction path.
        ######################
        x = self.non_linearity(self.conv1x1(x))
        residual_list = []
        for d in range(self.depth):
            x = self.down_list[d](x)
            residual_list.append(x.clone())
            x = self.non_linearity(self.down_conn_list[d](x))
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=0.5,
                                                mode='bilinear',
                                                align_corners=False)
        x = self.bottleneck(x)

        ######################
        # Expansion path.
        ######################
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

        if return_grad:
            vec_field_gradients = 0
            for i in range(len(self.ode_list)):
                vec_field_gradients += self.ode_list[i].vec_grad()
            return output, vec_field_gradients.mean() / len(self.ode_list)
        else:
            return output
