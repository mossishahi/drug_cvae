from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

# from .encoder import Encoder
# from .decoder import Decoder
from .loss import mse
from .base import CVAELatentsModelMixin, Decoder, Encoder
from ._utils import label_encoder

class gCVAE(nn.Module, CVAELatentsModelMixin):
    """
       ----------
       input_dim: Integer
            Number of input features 
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
    """

    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 hidden_layer_sizes: list = [256, 64],
                 latent_dim: int = 10,
                 recon_loss = 'mse',
                 dr_rate = 0.05,
                 use_bn :bool = False
                 ):
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        print("0------")
        print(recon_loss)
        assert recon_loss in ["mse"]  
        self.conditional = False
        print("\nINITIALIZING NEW NETWORK..............")
        print("---------")
        self.dr_rate = dr_rate
        print(self.dr_rate)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_conditions = len(conditions)
        self.conditions = conditions
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))}
        self.cell_type_encoder = None
        self.recon_loss = recon_loss
        self.use_bn = use_bn

        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        self.hidden_layer_sizes = hidden_layer_sizes
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)
        decoder_layer_sizes = self.hidden_layer_sizes.copy()
        decoder_layer_sizes.reverse()
        decoder_layer_sizes.append(self.input_dim)

        self.encoder = Encoder(encoder_layer_sizes,
                               self.latent_dim,
                               self.conditional,
                               self.n_conditions,
                               self.use_bn,
                               self.use_dr,
                               self.dr_rate,
                               )
        self.decoder = Decoder(decoder_layer_sizes,
                               self.latent_dim,
                               self.recon_loss,
                               self.conditional,
                               self.n_conditions,
                               self.use_bn,
                               self.use_dr,
                               self.dr_rate,
                               )

    def forward(self, x=None, batch=None, sizefactor=None, labeled=None):
        x_log = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_log = x
        z1_mean, z1_log_var = self.encoder(x_log, batch)
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        
        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = kl_divergence(
            Normal(z1_mean, torch.sqrt(z1_var)),
            Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var))
        ).sum(dim=1).mean()
        print("shapes", recon_loss.shape, kl_div.shape)
        return recon_loss, kl_div


    def forward(self, batch):

        x, c = batch["rna"], batch["coord"]

        # Encode rna
        x = self.rna_encoder(x, c)
        mu_rna, logvar_rna = self.encoder(x)
        z_rna = self.sampling(mu_rna, logvar_rna)

        rna_outputs = []
        f_rna = self.decoder(z_rna)
        rna_outputs.append(self.rna_decoder(f_rna, c))
        return {
            "recon_rna": rna_outputs,
            "z_rna": z_rna,
        }
