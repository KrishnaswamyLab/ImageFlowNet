from base import BaseNetwork
import math
import torch
from torch import nn


class UNet(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A U-Net model.

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
        super(UNet, self).__init__()

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

        time_embed_dim = n_f * 16
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.
        '''

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

        # Time embedding through feature space addition.
        assert x.shape[0] == 1 and x.shape[1] == self.time_embed_dim
        t_emb = self.time_embed(timestep_embedding(t, dim=self.time_embed_dim))
        t_emb = t_emb[:, :, None, None].repeat((1, 1, x.shape[2], x.shape[3]))
        x = x + t_emb

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


class ResUNet(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        A Residual U-Net model.

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
        super(ResUNet, self).__init__()

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

        time_embed_dim = n_f * 16
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.
        '''

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

        # Time embedding through feature space addition.
        assert x.shape[0] == 1 and x.shape[1] == self.time_embed_dim
        t_emb = self.time_embed(timestep_embedding(t, dim=self.time_embed_dim))
        t_emb = t_emb[:, :, None, None].repeat((1, 1, x.shape[2], x.shape[3]))
        x = x + t_emb

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
