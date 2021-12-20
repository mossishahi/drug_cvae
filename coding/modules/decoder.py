import torch.nn as nn
import torch
from .condlayers import CondLayers
from ._utils import one_hot_encoder

class Decoder(nn.Module):
    """ScArches Decoder class. Constructs the decoder sub-network of TRVAE or CVAE networks. It will transform the
       constructed latent space to the previous space of data with n_dimensions = x_dimension.
       Parameters
       ----------
       layer_sizes: List
            List of hidden and last layer sizes
       latent_dim: Integer
            Bottleneck layer (z)  size.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
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
                 recon_loss: str,
                 use_dr: bool,
                 use_bn: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.n_classes = 0
        if num_classes is not None:
            self.n_classes = num_classes
        layer_sizes = [latent_dim] + layer_sizes
        print("Decoder Architecture:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print("\tFirst Layer in, out and cond: ", layer_sizes[0], layer_sizes[1], self.n_classes)
        # self.FirstL.add_module(name="L0", module=CondLayers(layer_sizes[0], layer_sizes[1], self.n_classes, bias=False))
        if use_bn:
            self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    if use_bn:
                        self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.ReLU())
                    if self.use_dr:
                        self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dr_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU())

    def forward(self, z, batch=None):
        # Add Condition Labels to Decoder Input
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=self.n_classes)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)

        # Compute Hidden Output
        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent

        # Compute Decoder Output
        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent