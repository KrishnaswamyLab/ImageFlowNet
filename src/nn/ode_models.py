import os

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint


class ConvODEResUNet(nn.Module):

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 num_filters: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 augment_dim: int = 0,
                 time_dependent: bool = False,
                 non_linearity: str = 'relu',
                 tol: float = 1e-3,
                 adjoint=False,
                 max_num_steps: int = 1000):
        '''
        A Residual U-Net model with the bottleneck layer built using a ConvODEBlock.
        Partially inspired by https://github.com/DIAGNijmegen/neural-odes-segmentation

        Parameters
        ----------
        device: torch.device
        num_filters : int
            Number of convolutional filters.
        in_channels: int
            Number of input image channels.
        out_channels: int
            Number of output image channels.
        augment_dim: int
            Number of augmentation channels to add. If 0 does not augment ODE.
        time_dependent : bool
            If True adds time as input, making ODE time dependent.
        non_linearity : string
            One of 'relu' and 'softplus'
        tol: float
            Error tolerance.
        adjoint: bool
            If True calculates gradient with adjoint method, otherwise
            backpropagates directly through operations of ODE solver.
        max_num_steps: int
            Max number of steps in ODE solver.
        '''
        super(ConvODEResUNet, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.tol = tol
        self.adjoint = adjoint
        self.max_num_steps = max_num_steps
        self.non_linearity_str = non_linearity
        if self.non_linearity_str == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif self.non_linearity_str == 'softplus':
            self.non_linearity = nn.Softplus()

        n_f = num_filters  # shorthand for simplicity

        self.conv1x1 = nn.Conv2d(in_channels, n_f, 1, 1)
        self.conv_down_1 = ResConvBlock(n_f)
        self.conv_down_1_2 = nn.Conv2d(n_f, n_f * 2, 1, 1)
        self.conv_down_2 = ResConvBlock(n_f * 2)
        self.conv_down_2_3 = nn.Conv2d(n_f * 2, n_f * 4, 1, 1)
        self.conv_down_3 = ResConvBlock(n_f * 4)
        self.conv_down_3_4 = nn.Conv2d(n_f * 4, n_f * 8, 1, 1)
        self.conv_down_4 = ResConvBlock(n_f * 8)
        self.conv_down_4_embed = nn.Conv2d(n_f * 8, n_f * 16, 1, 1)

        self.odeblock_embedding = self._make_ode_block(prev_channels=n_f * 16,
                                                       curr_channels=n_f * 16)

        self.conv_up_embed_4 = nn.Conv2d(n_f * 16 + n_f * 8 + self.augment_dim,
                                         n_f * 8, 1, 1)
        self.conv_up_4 = ResUpConvBlock(n_f * 8)
        self.conv_up_4_3 = nn.Conv2d(n_f * 8 + n_f * 4, n_f * 4, 1, 1)
        self.conv_up_3 = ResUpConvBlock(n_f * 4)
        self.conv_up_3_2 = nn.Conv2d(n_f * 4 + n_f * 2, n_f * 2, 1, 1)
        self.conv_up_2 = ResUpConvBlock(n_f * 2)
        self.conv_up_2_1 = nn.Conv2d(n_f * 2 + n_f, n_f, 1, 1)
        self.conv_up_1 = ResUpConvBlock(n_f)

        self.out_layer = nn.Conv2d(n_f, out_channels, 1)

    def _make_ode_block(self, prev_channels: int, curr_channels: int):
        ode_func = ConvODEFunc(device=self.device,
                               prev_channels=prev_channels,
                               num_filters=curr_channels,
                               augment_dim=self.augment_dim,
                               time_dependent=self.time_dependent,
                               non_linearity=self.non_linearity_str)
        return ODEBlock(device=self.device,
                        odefunc=ode_func,
                        is_conv=True,
                        tol=self.tol,
                        adjoint=self.adjoint,
                        max_num_steps=self.max_num_steps)

    def forward(self, x: torch.Tensor, eval_times: torch.Tensor = None):
        '''
        In the current implementation, we assume `eval_times` is
        an array of 2 elements, containing [t_begin, t_end].

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

        # Assuming `eval_times` contains [t_start, t_end],
        # we only return the features at t_end.
        x = self.odeblock_embedding(x, eval_times=eval_times)[-1]

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

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        return


class ResUNet(nn.Module):

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

    def forward(self, x: torch.Tensor, eval_times: None):
        '''
        `interpolate` is used as a drop-in replacement for MaxPool2d.

        `eval_times` is a dummy input argument.
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

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        return


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


# =============================================================================================== #
# Below this line are modules adapted from https://github.com/EmilienDupont/augmented-neural-odes #
# =============================================================================================== #


class ODEFunc(nn.Module):
    """MLP modeling the derivative of ODE system.
    Parameters
    ----------
    device : torch.device
    data_dim : int
        Dimension of data.
    hidden_dim : int
        Dimension of hidden layers.
    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    """

    def __init__(self,
                 device,
                 data_dim,
                 hidden_dim,
                 augment_dim=0,
                 time_dependent=False,
                 non_linearity='relu'):
        super(ODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.num_filterse = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_dim)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).
        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        self.num_filterse += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        return out


class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.
    Parameters
    ----------
    device : torch.device
    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.
    is_conv : bool
        If True, treats odefunc as a convolutional model.
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """

    def __init__(self,
                 device,
                 odefunc,
                 is_conv=False,
                 tol=1e-3,
                 adjoint=False,
                 max_num_steps=1000):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol
        # Maximum number of steps for ODE solver
        self.max_num_steps = max_num_steps

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.num_filterse = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.odefunc.augment_dim, height,
                                  width).to(self.device)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = torch.cat([x, aug], 1)
            else:
                # Add augmentation
                aug = torch.zeros(x.shape[0],
                                  self.odefunc.augment_dim).to(self.device)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:
            out = odeint_adjoint(self.odefunc,
                                 x_aug,
                                 integration_time,
                                 rtol=self.tol,
                                 atol=self.tol,
                                 method='dopri5',
                                 options={'max_num_steps': self.max_num_steps})
        else:
            out = odeint(self.odefunc,
                         x_aug,
                         integration_time,
                         rtol=self.tol,
                         atol=self.tol,
                         method='dopri5',
                         options={'max_num_steps': self.max_num_steps})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)


class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.
    Parameters
    ----------
    device : torch.device
    prev_channels: int
        Number of channels from the previous layer which will be kept after this module.
    num_filters : int
        Number of convolutional filters.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    """

    def __init__(self,
                 device,
                 prev_channels,
                 num_filters,
                 augment_dim=0,
                 time_dependent=False,
                 non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.num_filterse = 0  # Number of function evaluations
        self.prev_channels = prev_channels
        self.prev_channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.prev_channels,
                                    self.num_filters,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
            self.conv2 = Conv2dTime(self.num_filters,
                                    self.num_filters,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
            self.conv3 = Conv2dTime(self.num_filters,
                                    self.prev_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        else:
            self.conv1 = nn.Conv2d(self.prev_channels,
                                   self.num_filters,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
            self.conv2 = nn.Conv2d(self.num_filters,
                                   self.num_filters,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.conv3 = nn.Conv2d(self.num_filters,
                                   self.prev_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.
        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.num_filterse += 1
        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            out = self.conv1(x)
            out = self.non_linearity(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
            out = self.conv3(out)
        return out
