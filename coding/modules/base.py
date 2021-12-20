import torch
import numpy as np
from torch.distributions import Normal
from ._utils import compute_out_size
from .activations import ACTIVATIONS

from torch import nn
import torch


class CVAELatentsModelMixin:
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.
           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x, c=None, mean=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           mean: boolean
           Returns
           -------
           Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean
        return latent

    def get_y(self, x, c=None):
        """Map `x` in to the y dimension (First Layer of Decoder). This function will feed data in encoder  and return
           y for each sample in data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           Returns
           -------
           Returns Torch Tensor containing output of first decoder layer.
        """

        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        output = self.decoder(latent, c)
        return output[-1]



def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """
    Convolution 1x1 layer.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups, bias=bias
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """
    Convolution 3x3 layer.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=False,
        bn_eps=1e-5,
        activate=True,
    ):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels, out_channels, stride=1, padding=0, groups=1, bias=False, bn_eps=1e-5, activate=True):
    """
    1x1 version of the standard convolution block.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        bn_eps=bn_eps,
        activate=activate,
    )


def conv3x3_block(
    in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False, bn_eps=1e-5, activate=True
):
    """
    3x3 version of the standard convolution block.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        bn_eps=bn_eps,
        activate=activate,
    )


class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks', see https://arxiv.org/abs/1709.01507.

    Taken https://github.com/moskomule/senet.pytorch/blob/8cb2669fec6fa344481726f9199aa611f08c3fbd/senet/se_module.py
    """

    def __init__(self, channel, reduction=16, pooling="avg"):
        super(SELayer, self).__init__()

        if pooling == "max":
            AdaptivePool2d = nn.AdaptiveMaxPool2d
        elif pooling == "avg":
            AdaptivePool2d = nn.AdaptiveAvgPool2d
        else:
            raise ValueError("Pooling has to be one of 'avg' or 'max'.")

        self.pool = AdaptivePool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResUnit(nn.Module):
    """
    SE-ResNet unit.

    Taken from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        pooling="max",
    ):
        super(SEResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.se = SELayer(channel=out_channels, pooling=pooling)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels, out_channels=out_channels, stride=stride, activate=False
            )
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class CNNEncoder(nn.Module):
    """
    Image encoder. Encode the image to latent space representation using SEResNet units.

    Parts from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models
    """

    def __init__(
        self,
        channels: list,
        init_block_channels: int = 64,
        in_channels: int = 3,
        in_size: tuple = (64, 64),
        h_dim: int = 256,
        pooling: str = "avg",
    ):
        super().__init__()

        self.channels = channels
        self.init_block_channels = init_block_channels
        self.in_channels = in_channels
        self.in_size = in_size
        self.h_dim = h_dim

        if in_size[0] != in_size[1]:
            raise Warning("Image is not quadratic.")
        self.out_size = in_size[0]

        if pooling == "max":
            Pool2d = nn.MaxPool2d
        elif pooling == "avg":
            Pool2d = nn.AvgPool2d
        else:
            raise ValueError("Pooling has to be one of 'avg' or 'max'.")

        self.init_layer = nn.Conv2d(in_channels, init_block_channels, kernel_size=7, stride=2, padding=3)
        self.out_size = compute_out_size(self.out_size, kernel_size=7, stride=2, padding=3)
        self.init_pool = Pool2d(kernel_size=2, stride=2)
        self.out_size = compute_out_size(self.out_size, kernel_size=2, stride=2, padding=0)

        self.features = nn.Sequential()
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module(
                    "unit{}".format(j + 1),
                    SEResUnit(in_channels=in_channels, out_channels=out_channels, stride=stride, pooling=pooling),
                )
                self.out_size = compute_out_size(self.out_size, kernel_size=3, stride=stride, padding=1)
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.final_pool = Pool2d(kernel_size=2, stride=2)
        self.out_size = compute_out_size(self.out_size, kernel_size=2, stride=2, padding=0)
        self.linear = nn.Linear(in_channels * self.out_size * self.out_size, h_dim)

    def forward(self, x):
        x = self.init_layer(x)
        x = self.init_pool(x)
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNNDecoder(nn.Module):
    """
    Image decoder. Decode the image from latent space representation using SEResNet units.

    Parts from https://github.com/osmr/imgclsmob/tree/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models
    """

    def __init__(
        self,
        channels: list,
        final_block_channels: int = 64,
        in_channels: int = 512,
        out_channels: int = 3,
        in_size: tuple = (1, 1),
        h_dim: int = 256,
        pooling: str = "avg",
    ):
        super().__init__()

        self.channels = channels
        self.final_block_channels = final_block_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.h_dim = h_dim

        self.linear = nn.Linear(self.h_dim, self.in_channels * self.in_size[0] * self.in_size[1])
        self.init_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.features = nn.Sequential()
        in_channels = self.in_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    stage.add_module("upsample", nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                stage.add_module(
                    "unit{}".format(j + 1),
                    SEResUnit(in_channels=in_channels, out_channels=out_channels, stride=1, pooling=pooling),
                )
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.last_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SEResUnit(in_channels=in_channels, out_channels=final_block_channels, stride=1),
            SEResUnit(in_channels=final_block_channels, out_channels=final_block_channels, stride=1),
            nn.Conv2d(final_block_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.in_channels, self.in_size[0], self.in_size[1])
        x = self.init_upsample(x)
        x = self.features(x)
        x = self.last_layer(x)
        return x

class Encoder(nn.Module):
    """RNA VAE encoder."""

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int = 100,
        conditional: bool = False,
        num_classes: int = 0,
        use_bn: bool = True,
        use_dr: bool = True,
        dr_rate: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.conditional = conditional
        self.num_classes = num_classes
        if self.conditional:
            layer_sizes[0] += self.num_classes

        if len(self.layer_sizes) > 1:
            self.FC = FullyConnected(layer_sizes, use_bn, use_dr, dr_rate, activation)

        self.linear_means = nn.Linear(layer_sizes[-1], latent_dim)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)
        if len(self.layer_sizes) > 1:
            x = self.FC(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class FullyConnected(nn.Module):
    """Fully connected layer."""

    def __init__(
        self,
        layer_sizes: list,
        use_bn: bool = True,
        use_dr: bool = True,
        dr_rate: float = 0.2,
        activation: str = "relu",
        conditional: bool = False,
        num_classes: int = 0,
    ):
        super().__init__()

        self.conditional = conditional
        self.num_classes = num_classes
        if self.conditional:
            layer_sizes[0] += self.num_classes

        self.FC = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
            if use_bn:
                self.FC.add_module("B{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
            self.FC.add_module(name="A{:d}".format(i), module=ACTIVATIONS[activation])
            if use_dr:
                self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)
        x = self.FC(x)
        return x


class Decoder(nn.Module):
    """RNA decoder using MSE, NB or ZINB output distribution."""

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int = 100,
        recon_loss: str = "mse",
        conditional: bool = False,
        num_classes: int = 0,
        use_bn: bool = True,
        use_dr: bool = True,
        dr_rate: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.recon_loss = recon_loss
        self.conditional = conditional
        self.num_classes = num_classes

        if self.conditional:
            input_size = latent_dim + num_classes
        else:
            input_size = latent_dim

        self.FC = FullyConnected([input_size] + layer_sizes[:-1], use_bn, use_dr, dr_rate, activation)

        if self.recon_loss == "mse":
            self.output_layer = nn.Sequential()
            self.output_layer.add_module(name="outputL", module=nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.output_layer.add_module(name="outputA", module=nn.ReLU())
        if self.recon_loss == "zinb":
            self.mean_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
            self.disp_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
            self.dropout_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Sigmoid())
        if self.recon_loss == "nb":
            self.mean_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
            self.disp_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)
        x = self.FC(z)
        if self.recon_loss == "mse":
            output = self.output_layer(x)
            return output
        if self.recon_loss == "zinb":
            # Parameters for ZINB
            dec_mean_gamma = self.mean_decoder(x)
            dec_disp = self.disp_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            dec_disp = dec_disp.exp().add(1).log().add(1e-3)
            dec_mean_gamma = dec_mean_gamma.exp().add(1).log().add(1e-4)
            return dec_mean_gamma, dec_disp, dec_dropout
        if self.recon_loss == "nb":
            # Parameters for NB
            dec_mean_gamma = self.mean_decoder(x)
            dec_disp = self.disp_decoder(x)
            dec_disp = dec_disp.exp().add(1).log().add(1e-3)
            dec_mean_gamma = dec_mean_gamma.exp().add(1).log().add(1e-4)
            return dec_mean_gamma, dec_disp


class DataLoader:
    def __init__(self) -> None:
        pass


class annDataLoader(DataLoader):
    def __init__(self,
                anndata,
                conditions) -> None:
        super().__init__()
        self.anndata = anndata
        self.conditions = conditions
    
    def __getitem__(self, index):
        cell = self.anndata.X.A[index, :]
        condition = self.anndata.obs.loc[:,self.conditions].iloc[index]
        return torch.cat((cell, condition))

