import torch.nn as nn
import torch
from ._utils import one_hot_encoder
from .condlayers import CondLayers


class Encoder(nn.Module):
    """ScArches Encoder class. Constructs the encoder sub-network of TRVAE and CVAE. It will transform primary space
       input to means and log. variances of latent space with n_dimensions = z_dimension.
       Parameters
       ----------
       layer_sizes: List
            List of first and hidden layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       use_dr: Boolean
            If `True` dropout will applied to layers.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropput will be applied.
       num_classes: Integer
            Number of classes (conditions) the data contain. if `None` the model will be a normal VAE instead of
            conditional VAE.
    """

    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_dr: bool,
                 use_bn: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        self.FC = None
        print(layer_sizes, "<<<.")
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out and cond:",
                          in_size, out_size, self.n_classes)
                    print("classs ", self.n_classes)
                    self.FC.add_module(name="L{:d}".format(i), module=CondLayers(in_size, out_size, self.n_classes, bias=True))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(
                        i), module=nn.Linear(in_size, out_size, bias=True))
                # if use_bn:
                #     self.FC.add_module("N{:d}".format(
                #         i), module=nn.BatchNorm1d(out_size, affine=True))

                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(name="D{:d}".format(
                        i), module=nn.Dropout(p=dr_rate))
        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, batch=None):
        print(x.shape, "shits")
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            print("here")
            x = self.FC(x)
        means = self.mean_encoder(x)
        print("oh! shit", means)
        log_vars = self.log_var_encoder(x)
        return means, log_vars
