import os
import torch
import torch.nn as nn
import inferno.extensions.layers.sampling as sampling
from inferno.extensions.layers.convolutional import Conv3D
from inferno.extensions.models.unet import UNetBase
from inferno.extensions.layers.identity import Identity
from inferno.trainers import Trainer


class GroupNormConv3d(nn.Module):
    default_group_size = 32

    def __init__(self, in_channels, out_channels, kernel_size,
                 depthwise=False, pad=True):
        super().__init__()
        # self.group_norm = nn.GroupNorm(min(in_channels,
        #                                    self.default_group_size),
        #                                in_channels)
        # parameters for the convolution
        padding = (kernel_size - 1) // 2 if pad else 0
        groups = in_channels if depthwise else 1
        assert groups <= out_channels, "%i, %i" % (groups, out_channels)

        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, groups=groups,
                              padding=padding)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x = self.group_norm(x)
        x = self.conv(x)
        return self.activation(x)


def get_activation(final_activation):
    # get the final output and activation activation
    if isinstance(final_activation, str):
        activation = getattr(nn, final_activation)()
    elif isinstance(final_activation, nn.Module):
        activation = final_activation
    elif final_activation is None:
        activation = None
    else:
        raise NotImplementedError("Activation of type %s is not supported" % type(final_activation))
    return activation


class UNetAnisotropic(UNetBase):
    """ UNet that allows for anisotropic scaling
    """
    def __init__(self, in_channels, scale_factors,
                 out_channels=None, initial_features=16, gain=3,
                 final_activation=None, p_dropout=None):

        # convolutional types for inner convolutions and output convolutions
        # self.default_conv = ConvELU3D
        self.default_conv = GroupNormConv3d
        last_conv = Conv3D

        assert isinstance(scale_factors, (list, tuple))
        assert all((isinstance(sf, (int, list, tuple)) for sf in scale_factors))
        self.scale_factors = scale_factors
        depth = len(scale_factors)
        # init the base class
        super(UNetAnisotropic, self).__init__(in_channels=initial_features, dim=3,
                                              depth=depth, gain=gain, p_dropout=p_dropout)
        # initial conv layer to go from the number of input channels,
        # which are defined by the data
        # (usually 1 or 3) to the initial number of feature maps
        self._initial_conv = self.default_conv(in_channels, initial_features, 3,
                                               depthwise=False)
        activation = get_activation(final_activation)

        # if out-channels are none, we return the vanailla u-net output
        # (regardless of the activation)
        if out_channels is None:
            self.out_channels = initial_features
            self._output = None
        elif activation is None:
            self.out_channels = int(out_channels)
            self._output = last_conv(initial_features, self.out_channels, 1)
        else:
            self.out_channels = int(out_channels)
            self._output = nn.Sequential(last_conv(initial_features, self.out_channels, 1),
                                         activation)

    def forward(self, input):
        x = self._initial_conv(input)
        x = super(UNetAnisotropic, self).forward(x)
        if self._output is None:
            return x
        else:
            return self._output(x)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        # is this the first convolutional block?
        first = (part == 'down' and index == 0)
        last = (part == 'up' and index == 0)

        # depthwise = part == 'down'
        depthwise = False

        # if this is the first conv block, we just need
        # a single convolution, because we have the `_initial_conv` already
        if first:
            conv = self.default_conv(in_channels, out_channels, 3,
                                     depthwise=depthwise)
        # in the last layer, we might have 'skip_last_up' set and use identity
        elif last:
            if getattr(self, 'skip_last_up', False):
                conv = Identity
            else:
                conv = nn.Sequential(self.default_conv(in_channels, out_channels,
                                                       3, depthwise=depthwise),
                                     self.default_conv(out_channels, out_channels,
                                                       3, depthwise=depthwise))
        else:
            conv = nn.Sequential(self.default_conv(in_channels, out_channels,
                                                   3, depthwise=depthwise),
                                 self.default_conv(out_channels, out_channels,
                                                   3, depthwise=depthwise))
        return conv, False

    def downsample_op_factory(self, index):
        sf = self.scale_factors[index]
        if isinstance(sf, int):
            pooler = nn.MaxPool3d(kernel_size=sf,
                                  stride=sf)
        else:
            assert len(sf) == 3
            assert sf[0] == 1
            assert sf[1] == sf[2]
            pooler = sampling.AnisotropicPool(sf[1])
        return pooler

    def upsample_op_factory(self, index):
        sf = self.scale_factors[index]
        sampler = sampling.Upsample(scale_factor=sf)
        return sampler


def save_best_model(project_directory):
    trainer = Trainer().load(from_directory=os.path.join(project_directory, "Weights"),
                             best=True, map_location='cpu')

    # save the model
    model = trainer.model
    save_path = os.path.join(project_directory, "Weights", "best_model.nn")
    torch.save(model, save_path)

    # save the state dict
    save_path = os.path.join(project_directory, "Weights", "best_model.state")
    torch.save(model.state_dict(), save_path)
