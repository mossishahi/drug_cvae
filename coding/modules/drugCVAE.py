import inspect
import os
import torch
import pickle
import numpy as np

from anndata import AnnData, read
from copy import deepcopy
from typing import Optional, Union
from .trainer import CVAETrainer
from .gCVAE import gCVAE
from .trainers import trVAETrainer
from .dCVAE import dCVAE
import pytorch_lightning as pl

class drugCVAE:
    """Model for scArches class. This class contains the implementation of Conditional Variational Auto-encoder.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       use_mmd: Boolean
            If 'True' an additional MMD loss will be calculated on the latent dim. 'z' or the first decoder layer 'y'.
       mmd_on: String
            Choose on which layer MMD loss will be calculated on if 'use_mmd=True': 'z' for latent dim or 'y' for first
            decoder layer.
       mmd_boundary: Integer or None
            Choose on how many conditions the MMD loss should be calculated on. If 'None' MMD will be calculated on all
            conditions.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
       beta: Float
            Scaling Factor for MMD loss
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """

    def __init__(
        self,
        adata: AnnData,
        layers_args,
        conditions: Optional[list] = None,
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        recon_loss = 'mse',
    ):
        self.adata = adata
        #name of the columns in adata.obs which should be used as Conditions
        
        self.input_size = layers_args["input_size"]
        self.rna_layer = layers_args["rna_layers"]
        self.latent_dim = layers_args["latent_dim"]
        self.h_dim = layers_args["h_dim"]
        self.conditions_ = conditions
        self.dr_rate_ = dr_rate
        self.recon_loss_ = recon_loss
        self.num_conditions = len(self.conditions_)

        # print(self.input_dim)
        self.model = dCVAE(
            layers_args,
            self.num_conditions,
            self.recon_loss_,
            self.dr_rate_
        )

        self.is_trained_ = False

        self.trainer = None
     

    def train(
        self,
        dataset,
    ):
        training_args = {
          "n_epochs": 50,
          "lr": 1e-3,
          "final_kl_weight": 1.0,
          "weight_decay": 0.04,
          }
        self.training_module = CVAETrainer(
            self.model,
            training_args,
            )
        self.cvae_trainer = pl.Trainer(    
                          max_epochs=self.training_module.n_epochs,
                          log_every_n_steps=1,
                          terminate_on_nan=True,
                          progress_bar_refresh_rate=1e6,
                          checkpoint_callback=False,
                          )   
        self.cvae_trainer.fit(self.training_module, dataset)
        self.is_trained_ = True
        return self.cvae_trainer.logged_metrics

























