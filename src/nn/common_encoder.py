import torch
from .base import BaseNetwork


class Encoder(BaseNetwork):

    def __init__(self,
                 in_channels: int,
                 n_f: int,
                 depth: int,
                 conv_block: torch.nn.Module,
                 non_linearity: torch.nn.Module,
                 bilinear: bool = True):
        super().__init__()

        self.depth = depth
        self.non_linearity = non_linearity
        self.bilinear = bilinear

        self.conv1x1 = torch.nn.Conv2d(in_channels, n_f, 1, 1)
        self.down_list = torch.nn.ModuleList([])
        self.down_conn_list = torch.nn.ModuleList([])
        for d in range(self.depth):
            self.down_list.append(conv_block(n_f * 2 ** d))
            self.down_conn_list.append(torch.nn.Conv2d(n_f * 2 ** d, n_f * 2 ** (d + 1), 1, 1))

        self.bottleneck = conv_block(n_f * 2 ** self.depth)

        if not self.bilinear:
            self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.non_linearity(self.conv1x1(x))

        residual_list = []
        for d in range(self.depth):
            x = self.down_list[d](x)
            residual_list.append(x.clone())
            x = self.non_linearity(self.down_conn_list[d](x))
            if self.bilinear:
                x = torch.nn.functional.interpolate(x,
                                                    scale_factor=0.5,
                                                    mode='bilinear',
                                                    align_corners=False)
            else:
                x = self.pooling(x)

        x = self.bottleneck(x)

        return x, residual_list

    def freeze_weights(self) -> None:
        '''
        Freeze the weights and make them unchangable during training.
        '''
        for param in self.parameters():
            param.requires_grad = False

    def copy_weights(self, other_instance: torch.nn.Module) -> None:
        '''
        Copy the weights from a given instance
        '''
        with torch.no_grad():
            for this_param, other_param in zip(self.parameters(), other_instance.parameters()):
                this_param.data.copy_(other_param.data)
