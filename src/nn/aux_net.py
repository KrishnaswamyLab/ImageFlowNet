from .base import BaseNetwork
from .nn_utils import ConvBlock, UpConvBlock, ResConvBlock, ResUpConvBlock, ODEfunc, ODEBlock
import torch


class AuxNet(BaseNetwork):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 depth: int = 5,
                 use_residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 non_linearity: str = 'relu'):
        '''
        Auxiliary Network that performs discrimination and segmentation.

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

        self.conv1x1 = torch.nn.Conv2d(in_channels, n_f, 1, 1)

        if self.use_residual:
            conv_block = ResConvBlock
            upconv_block = ResUpConvBlock
        else:
            conv_block = ConvBlock
            upconv_block = UpConvBlock

        # This is for the encoder.
        self.down_list = torch.nn.ModuleList([])
        self.down_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            self.down_conn_list.append(torch.nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))

        self.bottleneck = conv_block(n_f * 2 ** self.depth)

        # This is for the segmentation head.
        self.up_list = torch.nn.ModuleList([])
        self.up_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.up_conn_list.append(torch.nn.Conv2d(n_f * 3 * 2 ** d, n_f * 2 ** d, 1, 1))
            self.up_list.append(upconv_block(n_f * 2 ** d))
        self.up_list = self.up_list[::-1]
        self.up_conn_list = self.up_conn_list[::-1]
        self.seg_head = torch.nn.ModuleList([
            torch.nn.Conv2d(n_f, out_channels, 1),
            torch.nn.Sigmoid(),
        ])

        # This is for the classification head
        self.cls_head = torch.nn.ModuleList([
            conv_block(n_f * 2 ** (self.depth + 1)),
            conv_block(n_f * 2 ** (self.depth + 1)),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Linear(n_f * 2 ** (self.depth + 1), 1)
        ])

    def forward_seg(self, x: torch.Tensor):
        '''
        Forward through the segmentation path.
        '''

        x = self.non_linearity(self.conv1x1(x))

        residual_list = []
        for d in range(self.depth):
            residual_list.append(x.copy())
            x = self.down_list[d](x)
            x = self.non_linearity(self.down_conn_list[d](x))
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=0.5,
                                                mode='bilinear',
                                                align_corners=False)
        x = self.bottleneck(x)

        for d in range(self.depth):
            x = torch.nn.functional.interpolate(x,
                                                scale_factor=2,
                                                mode='bilinear',
                                                align_corners=False)
            x = torch.cat([x, residual_list.pop(-1)], dim=1)
            x = self.non_linearity(self.up_conn_list[d](x))
            x = self.up_list[d](x)

        output = self.seg_head(x)

        return output

    def forward_cls(self, x1: torch.Tensor, x2: torch.Tensor):
        '''
        Forward through the classification path.
        '''

        x1 = self.non_linearity(self.conv1x1(x1))
        for d in range(self.depth):
            x1 = self.down_list[d](x1)
            x1 = self.non_linearity(self.down_conn_list[d](x1))
            x1 = torch.nn.functional.interpolate(x1,
                                                 scale_factor=0.5,
                                                 mode='bilinear',
                                                 align_corners=False)
        x1 = self.bottleneck(x1)

        x2 = self.non_linearity(self.conv1x1(x2))
        for d in range(self.depth):
            x2 = self.down_list[d](x2)
            x2 = self.non_linearity(self.down_conn_list[d](x2))
            x2 = torch.nn.functional.interpolate(x2,
                                                 scale_factor=0.5,
                                                 mode='bilinear',
                                                 align_corners=False)
        x2 = self.bottleneck(x2)

        x = torch.cat([x1, x2], dim=1)
        output = self.cls_head(x)

        return output

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Please use `forward_seg` or `forward_cls` instead.')
