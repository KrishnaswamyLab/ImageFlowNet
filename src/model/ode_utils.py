'''
Adapted from https://github.com/DIAGNijmegen/neural-odes-segmentation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint, odeint_adjoint

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Utility class that wraps odeint and odeint_adjoint.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)


class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Conv2d module where time gets concatenated as a feature map.
        Makes ODE func aware of the current time step.
        """
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)

def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()

class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)

class ConvODEFunc(nn.Module):
    def __init__(self, nf, time_dependent=False, non_linearity='relu'):
        """
        Block for ConvODEUNet

        Args:
            nf (int): number of filters for the conv layers
            time_dependent (bool): whether to concat the time as a feature map before the convs
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvODEFunc, self).__init__()
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations

        if time_dependent:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = Conv2dTime(nf, nf, kernel_size=3, stride=1, padding=1)
        else:
            self.norm1 = nn.InstanceNorm2d(nf)
            self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(nf)
            self.conv2 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            out = self.norm1(x)
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
        else:
            out = self.norm1(x)
            out = self.conv1(out)
            out = self.non_linearity(out)
            out = self.norm2(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
        return out

class ConvResFunc(nn.Module):
    def __init__(self, num_filters, non_linearity='relu'):
        """
        Block for ConvResUNet

        Args:
            num_filters (int): number of filters for the conv layers
            non_linearity (str): which non_linearity to use (for options see get_nonlinearity)
        """
        super(ConvResFunc, self).__init__()

        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(2, num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(x)
        out = self.non_linearity(out)
        out = self.norm(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        out = x + out
        return out

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        Block for LevelBlock

        Args:
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
        """
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3), nn.ReLU(inplace=True),
        )

class LevelBlock(nn.Module):
    def __init__(self, depth, total_depth, in_channels, out_channels):
        """
        Block for UNet

        Args:
            depth (int): current depth of blocks (starts with total_depth: n,...,0)
            total_depth (int): total_depth of U-Net
            in_channels (int): number of input filters for first conv layer
            out_channels (int): number of output filters for the last layer
        """
        super(LevelBlock, self).__init__()
        self.depth = depth
        self.total_depth = total_depth
        if depth > 1:
            self.encode = ConvBlock(in_channels, out_channels)
            self.down = nn.MaxPool2d(2, 2)
            self.next = LevelBlock(depth - 1, total_depth, out_channels, out_channels * 2)
            next_out = list(self.next.modules())[-2].out_channels
            self.up = nn.ConvTranspose2d(next_out, next_out // 2, 2, 2)
            self.decode = ConvBlock(next_out // 2 + out_channels, out_channels)
        else:
            self.embed = ConvBlock(in_channels, out_channels)

    def forward(self, inp):
        if self.depth > 1:
            first_x = self.encode(inp)
            x = self.down(first_x)
            x = self.next(x)
            x = self.up(x)

            # center crop
            i_h = first_x.shape[2]
            i_w = first_x.shape[3]

            total_crop = i_h - x.shape[2]
            crop_left_top = total_crop // 2
            crop_right_bottom = total_crop - crop_left_top

            cropped_input = first_x[:, :,
                                    crop_left_top:i_h - crop_right_bottom,
                                    crop_left_top:i_w - crop_right_bottom]
            x = torch.cat((cropped_input, x), dim=1)

            x = self.decode(x)
        else:
            x = self.embed(inp)

        return x
