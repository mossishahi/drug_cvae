from .trainer import CVAETrainer
import torch.nn as nn
import torch
import torch.optim as optim
import pytorch_lightning as pl

class trVAETrainer(CVAETrainer):
    """ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
       Trainer.
           Parameters
           ----------
           model: trVAE
                Number of input features (i.e. gene in case of scRNA-seq).
           adata: : `~anndata.AnnData`
                Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
                for 'mse' loss.
           condition_key: String
                column name of conditions in `adata.obs` data frame.
           cell_type_key: String
                column name of celltypes in `adata.obs` data frame.
           train_frac: Float
                Defines the fraction of data that is used for training and data that is used for validation.
           batch_size: Integer
                Defines the batch size that is used during each Iteration
           n_samples: Integer or None
                Defines how many samples are being used during each epoch. This should only be used if hardware resources
                are limited.
           clip_value: Float
                If the value is greater than 0, all gradients with an higher value will be clipped during training.
           weight decay: Float
                Defines the scaling factor for weight decay in the Adam optimizer.
           alpha_iter_anneal: Integer or None
                If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
                integer is reached.
           alpha_epoch_anneal: Integer or None
                If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every epoch until the input
                integer is reached.
           use_early_stopping: Boolean
                If 'True' the EarlyStopping class is being used for training to prevent overfitting.
           early_stopping_kwargs: Dict
                Passes custom Earlystopping parameters.
           use_stratified_sampling: Boolean
                If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
                iteration.
           use_stratified_split: Boolean
                If `True`, the train and validation data will be constructed in such a way that both have same distribution
                of conditions in the data.
           monitor: Boolean
                If `True', the progress of the training will be printed after each epoch.
           n_workers: Integer
                Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
           seed: Integer
                Define a specific random seed to get reproducable results.
        """
    def __init__(
            self,
            model,
            adata,
            **kwargs
    ):
        super().__init__(model, adata,**kwargs)

class dTrainer(pl.LightningModule):
    """Trainer class."""

    def __init__(
        self,
        model: nn.Module,
        training_args: dict,
    ):
        super().__init__()
        self.model = model
        self.n_epochs = training_args["n_epochs"]
        self.alpha_epoch_anneal = int(round(training_args["n_epochs"] * 0.3))
        self.final_kl_weight = training_args["final_kl_weight"]
        self.lr = training_args["lr"]
        self.weight_decay = training_args["weight_decay"]
        self.img_recon_loss_weight = training_args["img_recon_loss_weight"]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_dic = self.model.loss(batch, outputs, self.kl_weight, self.img_recon_loss_weight)
        self.log_dict(loss_dic)
        return loss_dic["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_dic = self.model.loss(batch, outputs, self.kl_weight, self.img_recon_loss_weight)
        self.log_dict({f"val_{k}": v for k, v in loss_dic.items()})

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_dic = self.model.loss(batch, outputs, self.kl_weight, self.img_recon_loss_weight)
        self.log_dict({f"test_{k}": v for k, v in loss_dic.items()})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        config = {"optimizer": optimizer}

        exponentiallr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)

        config.update(
            {"lr_scheduler": exponentiallr_scheduler, "monitor": "loss"},
        )

        return config

    @property
    def kl_weight(self):
        if self.alpha_epoch_anneal > 0:
            kl_weight = min(self.final_kl_weight, self.final_kl_weight * (self.current_epoch / self.alpha_epoch_anneal))
        else:
            kl_weight = self.final_kl_weight
        return kl_weight 