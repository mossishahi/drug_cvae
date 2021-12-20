import torch
import torch.nn as nn
from .base import CNNEncoder, FullyConnected, Encoder, Decoder, CNNDecoder
from .loss import mse, zinb, kl, nb, zinb


class BaseVAE(nn.Module):

    def __init__(self):
        super().__init__()

    def sampling(self, mu, log_var):
        """Re-Parametrization."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu).reshape(len(mu), -1)

class dCVAE(BaseVAE):

    def __init__(
        self,
        layers_args: dict,
        num_conditions: int = -1,
        use_batch_norm: bool = True,
        dropout_rate: float = 2e-1,
        rna_recon_loss: str = "mse",
        activation: str = "relu",
    ):
        super().__init__()


        self.input_size = layers_args["input_size"]
        self.rna_layers = layers_args["rna_layers"]
        self.latent_dim = layers_args["latent_dim"]
        self.h_dim = layers_args["h_dim"]
        self.num_conditions = num_conditions
        self.dr_rate_ = dropout_rate
        self.recon_loss_ = rna_recon_loss
        self.use_batch_norm = use_batch_norm
        self.rna_recon_loss = rna_recon_loss
        self.activation = activation

        self.num_conditions = num_conditions
        if self.num_conditions > 0:
            self.conditional = True
        else:
            self.conditional = False


        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.use_dropout = True
        else:
            self.use_dropout = False

        # prepare layers
        self.rna_encoder_layers = self.rna_layers
        self.rna_encoder_layers.insert(0, self.input_size)
        self.rna_decoder_layers = self.rna_encoder_layers[::-1]
        self.rna_encoder_layers.append(self.latent_dim)

        # self.hidden_encoder_layers = self.hidden_layers
        # self.hidden_encoder_layers.insert(0, self.h_dim)
        # self.hidden_decoder_layers = self.hidden_encoder_layers[::-1]
        # self.hidden_decoder_layers.insert(0, self.latent_dim)

        self.rna_encoder = Encoder(
            self.rna_encoder_layers,
            self.latent_dim,
            self.conditional,
            self.num_conditions,
            self.use_batch_norm,
            self.use_dropout,
            self.dropout_rate,
            self.activation,
        )
        self.rna_decoder = Decoder(
            self.rna_decoder_layers,
            self.h_dim,
            self.rna_recon_loss,
            False,
            -1,
            self.use_batch_norm,
            self.use_dropout,
            self.dropout_rate,
            self.activation,
        )
        # self.rna_hidden_encoder = Encoder(
        #     self.hidden_encoder_layers,
        #     self.latent_dim,
        #     False,
        #     -1,
        #     self.use_batch_norm,
        #     self.use_dropout,
        #     self.dropout_rate,
        #     self.activation,
        # )
        # self.rna_hidden_decoder = FullyConnected(
        #     self.hidden_decoder_layers,
        #     self.use_batch_norm,
        #     self.use_dropout,
        #     self.dropout_rate,
        #     self.activation,
        # )

    def forward(self, batch):

        x, c = batch["rna"], batch["coord"]

        # Encode rna
        mu_rna, logvar_rna = self.rna_encoder(x, c)
        # mu_rna, logvar_rna = self.rna_hidden_encoder(x)
        z_rna = self.sampling(mu_rna, logvar_rna)

        # Reconstruct 
        rna_outputs = []
        f_rna = self.rna_decoder(z_rna, c)
        # rna_outputs.append(self.rna_decoder(f_rna, c))

        return {
            "recon_rna": f_rna,
            "z_rna": z_rna,
            "z_mean":mu_rna,
            "z_log_var":logvar_rna
        }

    def loss(self, inputs, outputs, kl_weight, img_recon_loss_weight):

        # compute reconstruction loss for rna and image data
        recon_loss_rna = []
        recon_loss_img = []

        if self.rna_recon_loss == "mse":
            recon_loss_rna.append(mse(inputs["rna"], outputs["recon_rna"]))
        elif self.rna_recon_loss == "zinb":
            dec_mean_gamma, dec_disp, dec_dropout = outputs["recon_rna"]
            size_factor_view = (
                inputs["size_factors"].unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            )
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss_rna.append(zinb(inputs["raw"], dec_mean, dec_disp, dec_dropout, mean=False).sum())
        elif self.rna_recon_loss == "nb":
            dec_mean_gamma, dec_disp = outputs["recon_rna"]
            size_factor_view = (
                inputs["size_factors"].unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            )
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss_rna.append(nb(inputs["raw"], dec_mean, dec_disp, mean=False).sum())
        else:
            raise NotImplementedError("The loss function {} is not implemented!".format(self.rna_recon_loss))

        # recon_loss_rna = 0.5 * sum(recon_loss_rna)
        # compute kl losses
        kl_loss_rna = kl(outputs["z_mean"], outputs["z_log_var"])

        return {
            "loss": recon_loss_rna[0],
            "kl_loss_rna": kl_loss_rna,
        }

    def get_latent(self, x, i, c=None, as_tensor=False, return_all=False):

        # Encode rna
        mu_rna, logvar_rna = self.rna_encoder(x)
        x = self.sampling(mu_rna, logvar_rna)

        if as_tensor:
            return mu_rna
        else:
            return (
                mu_rna.detach().cpu().data.numpy()
            )