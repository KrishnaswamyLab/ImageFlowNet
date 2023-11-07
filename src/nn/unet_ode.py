from .base import BaseNetwork
import torch
from torch import nn
from torchdiffeq import odeint


class ConcatConv2d(nn.Module):

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
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
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


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = nn.GroupNorm(min(32, dim), dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = nn.GroupNorm(min(32, dim), dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, tolerance: float = 1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tolerance = tolerance

    def forward(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        out = odeint(self.odefunc,
                     x,
                     integration_time,
                     rtol=self.tolerance,
                     atol=self.tolerance)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class UNetODE(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A U-Net model with ODE.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        non_linearity : string
            One of 'relu' and 'softplus'
        '''
        super(UNetODE, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = nn.Softplus()

        n_f = num_filters  # shorthand

        self.conv1x1 = nn.Conv2d(in_channels, n_f, 1, 1)
        self.conv_down_1 = ConvBlock(n_f)
        self.conv_down_1_2 = nn.Conv2d(n_f, n_f * 2, 1, 1)
        self.conv_down_2 = ConvBlock(n_f * 2)
        self.conv_down_2_3 = nn.Conv2d(n_f * 2, n_f * 4, 1, 1)
        self.conv_down_3 = ConvBlock(n_f * 4)
        self.conv_down_3_4 = nn.Conv2d(n_f * 4, n_f * 8, 1, 1)
        self.conv_down_4 = ConvBlock(n_f * 8)
        self.conv_down_4_embed = nn.Conv2d(n_f * 8, n_f * 16, 1, 1)

        self.block_embedding = ConvBlock(n_f * 16)

        self.conv_up_embed_4 = nn.Conv2d(n_f * 16 + n_f * 8, n_f * 8, 1, 1)
        self.conv_up_4 = UpConvBlock(n_f * 8)
        self.conv_up_4_3 = nn.Conv2d(n_f * 8 + n_f * 4, n_f * 4, 1, 1)
        self.conv_up_3 = UpConvBlock(n_f * 4)
        self.conv_up_3_2 = nn.Conv2d(n_f * 4 + n_f * 2, n_f * 2, 1, 1)
        self.conv_up_2 = UpConvBlock(n_f * 2)
        self.conv_up_2_1 = nn.Conv2d(n_f * 2 + n_f, n_f, 1, 1)
        self.conv_up_1 = UpConvBlock(n_f)

        self.out_layer = nn.Conv2d(n_f, out_channels, 1)

        self.ode_block_embedding = ODEBlock(ODEfunc(dim=n_f * 16))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.

        Time embedding through ODE.
        '''

        assert x.shape[0] == 1

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        x = self.non_linearity(self.conv1x1(x))

        x_scale1 = self.conv_down_1(x)
        x = self.non_linearity(self.conv_down_1_2(x_scale1))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x_scale2 = self.conv_down_2(x)
        x = self.non_linearity(self.conv_down_2_3(x_scale2))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x_scale3 = self.conv_down_3(x)
        x = self.non_linearity(self.conv_down_3_4(x_scale3))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x_scale4 = self.conv_down_4(x)
        x = self.non_linearity(self.conv_down_4_embed(x_scale4))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x = self.block_embedding(x)
        if use_ode:
            x = self.ode_block_embedding(x, integration_time)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale4), dim=1)
        x = self.non_linearity(self.conv_up_embed_4(x))
        x = self.conv_up_4(x)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale3), dim=1)
        x = self.non_linearity(self.conv_up_4_3(x))
        x = self.conv_up_3(x)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale2), dim=1)
        x = self.non_linearity(self.conv_up_3_2(x))
        x = self.conv_up_2(x)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale1), dim=1)
        x = self.non_linearity(self.conv_up_2_1(x))
        x = self.conv_up_1(x)

        output = self.out_layer(x)

        return output


class ResUNetODE(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A Residual U-Net model with ODE.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        non_linearity : string
            One of 'relu' and 'softplus'
        '''
        super(ResUNetODE, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = nn.Softplus()

        n_f = num_filters  # shorthand

        self.conv1x1 = nn.Conv2d(in_channels, n_f, 1, 1)
        self.conv_down_1 = ResConvBlock(n_f)
        self.conv_down_1_2 = nn.Conv2d(n_f, n_f * 2, 1, 1)
        self.conv_down_2 = ResConvBlock(n_f * 2)
        self.conv_down_2_3 = nn.Conv2d(n_f * 2, n_f * 4, 1, 1)
        self.conv_down_3 = ResConvBlock(n_f * 4)
        self.conv_down_3_4 = nn.Conv2d(n_f * 4, n_f * 8, 1, 1)
        self.conv_down_4 = ResConvBlock(n_f * 8)
        self.conv_down_4_embed = nn.Conv2d(n_f * 8, n_f * 16, 1, 1)

        self.block_embedding = ResConvBlock(n_f * 16)

        self.conv_up_embed_4 = nn.Conv2d(n_f * 16 + n_f * 8, n_f * 8, 1, 1)
        self.conv_up_4 = ResUpConvBlock(n_f * 8)
        self.conv_up_4_3 = nn.Conv2d(n_f * 8 + n_f * 4, n_f * 4, 1, 1)
        self.conv_up_3 = ResUpConvBlock(n_f * 4)
        self.conv_up_3_2 = nn.Conv2d(n_f * 4 + n_f * 2, n_f * 2, 1, 1)
        self.conv_up_2 = ResUpConvBlock(n_f * 2)
        self.conv_up_2_1 = nn.Conv2d(n_f * 2 + n_f, n_f, 1, 1)
        self.conv_up_1 = ResUpConvBlock(n_f)

        self.out_layer = nn.Conv2d(n_f, out_channels, 1)

        self.ode_block_embedding = ODEBlock(ODEfunc(dim=n_f * 16))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.

        Time embedding through ODE.
        '''

        assert x.shape[0] == 1

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        x = self.non_linearity(self.conv1x1(x))

        x_scale1 = self.conv_down_1(x)
        x = self.non_linearity(self.conv_down_1_2(x_scale1))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x_scale2 = self.conv_down_2(x)
        x = self.non_linearity(self.conv_down_2_3(x_scale2))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x_scale3 = self.conv_down_3(x)
        x = self.non_linearity(self.conv_down_3_4(x_scale3))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x_scale4 = self.conv_down_4(x)
        x = self.non_linearity(self.conv_down_4_embed(x_scale4))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x = self.block_embedding(x)

        if use_ode:
            x = self.ode_block_embedding(x, integration_time)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale4), dim=1)
        x = self.non_linearity(self.conv_up_embed_4(x))
        x = self.conv_up_4(x)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale3), dim=1)
        x = self.non_linearity(self.conv_up_4_3(x))
        x = self.conv_up_3(x)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale2), dim=1)
        x = self.non_linearity(self.conv_up_3_2(x))
        x = self.conv_up_2(x)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale1), dim=1)
        x = self.non_linearity(self.conv_up_2_1(x))
        x = self.conv_up_1(x)

        output = self.out_layer(x)

        return output


class ShallowResUNetODE(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A Residual U-Net model with ODE.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        non_linearity : string
            One of 'relu' and 'softplus'
        '''
        super(ShallowResUNetODE, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = nn.Softplus()

        n_f = num_filters  # shorthand

        self.conv1x1 = nn.Conv2d(in_channels, n_f, 1, 1)
        self.conv_down_1 = ResConvBlock(n_f)
        self.conv_down_1_embed = nn.Conv2d(n_f, n_f * 2, 1, 1)

        self.block_embedding = ResConvBlock(n_f * 2)

        self.conv_up_embed_1 = nn.Conv2d(n_f * 2 + n_f, n_f, 1, 1)
        self.conv_up_1 = ResUpConvBlock(n_f)

        self.out_layer = nn.Conv2d(n_f, out_channels, 1)

        self.ode_block_embedding = ODEBlock(ODEfunc(dim=n_f * 2))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.

        Time embedding through ODE.
        '''

        assert x.shape[0] == 1

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        x = self.non_linearity(self.conv1x1(x))

        x_scale1 = self.conv_down_1(x)
        x = self.non_linearity(self.conv_down_1_embed(x_scale1))
        x = nn.functional.interpolate(x,
                                      scale_factor=0.5,
                                      mode='bilinear',
                                      align_corners=False)

        x = self.block_embedding(x)

        if use_ode:
            x = self.ode_block_embedding(x, integration_time)

        x = nn.functional.interpolate(x,
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False)
        x = torch.cat((x, x_scale1), dim=1)
        x = self.non_linearity(self.conv_up_embed_1(x))
        x = self.conv_up_1(x)

        output = self.out_layer(x)

        return output


class AutoEncoderODE(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 5,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        An AutoEncoder model with ODE.

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
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
            self.non_linearity = nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = nn.Softplus()

        n_f = num_filters  # shorthand

        self.conv1x1 = nn.Conv2d(in_channels, n_f, 1, 1)

        self.down_list = nn.ModuleList([])
        self.down_conn_list = nn.ModuleList([])
        self.up_list = nn.ModuleList([])
        self.up_conn_list = nn.ModuleList([])

        if self.use_residual:
            conv_block = ResConvBlock
            upconv_block = ResUpConvBlock
        else:
            conv_block = ConvBlock
            upconv_block = UpConvBlock

        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            self.down_conn_list.append(nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))
            self.up_conn_list.append(nn.Conv2d(n_f * 2 ** (d + 1), n_f * 2 ** d, 1, 1))
            self.up_list.append(upconv_block(n_f * 2 ** d))

        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]

        self.bottleneck = ResConvBlock(n_f * 2 ** self.depth)
        self.ode_bottleneck = ODEBlock(ODEfunc(dim=n_f * 2 ** self.depth))
        self.out_layer = nn.Conv2d(n_f, out_channels, 1)


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.

        Time embedding through ODE.
        '''

        assert x.shape[0] == 1

        # Skip ODE if no time difference.
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, t.item()]).float().to(t.device)

        x = self.non_linearity(self.conv1x1(x))

        for d in range(self.depth):
            x = self.down_list[d](x)
            x = self.non_linearity(self.down_conn_list[d](x))
            x = nn.functional.interpolate(x,
                                          scale_factor=0.5,
                                          mode='bilinear',
                                          align_corners=False)
        x = self.bottleneck(x)

        if use_ode:
            x = self.ode_bottleneck(x, integration_time)

        for d in range(self.depth):
            x = nn.functional.interpolate(x,
                                          scale_factor=2,
                                          mode='bilinear',
                                          align_corners=False)
            x = self.non_linearity(self.up_conn_list[d](x))
            x = self.up_list[d](x)

        output = self.out_layer(x)

        return output



class ConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.conv(x)


class ResConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.conv(x) + x


class UpConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.upconv(x)


class ResUpConvBlock(nn.Module):

    def __init__(self, num_filters):
        super(ResUpConvBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.InstanceNorm2d(num_filters),
            nn.Conv2d(num_filters,
                      num_filters,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True))

    def forward(self, x):
        return self.upconv(x) + x
