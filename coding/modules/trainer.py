import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
# from ._utils import custom_collate
# from ._dataloader import make_dataset
import pytorch_lightning as pl
import torch.optim as optim

class CVAETrainer(pl.LightningModule):
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_dic = self.model.loss(batch, outputs, self.kl_weight)
        self.log_dict(loss_dic)
        return loss_dic["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_dic = self.model.loss(batch, outputs, self.kl_weight)
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



    # def __init__(self,
    #              model,
    #              adata,
    #              conditions: list = None,
    #              batch_size: int = 128,
    #              alpha_epoch_anneal: int = None,
    #              alpha_kl: float = 1.,
    #              use_early_stopping: bool = False,
    #              reload_best: bool = True,
    #              early_stopping_kwargs: dict = None,
    #              **kwargs):

    #     self.adata = adata
    #     self.model = model
    #     self.conditions = conditions
    #     # self.cell_type_keys = cell_type_keys

    #     self.batch_size = batch_size
    #     self.alpha_epoch_anneal = alpha_epoch_anneal
    #     self.alpha_iter_anneal = kwargs.pop("alpha_iter_anneal", None)
    #     self.use_early_stopping = use_early_stopping
    #     self.reload_best = reload_best

    #     self.alpha_kl = alpha_kl

    #     early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else dict())

    #     self.n_samples = kwargs.pop("n_samples", None)
    #     self.train_frac = kwargs.pop("train_frac", 0.9)
    #     self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", False)

    #     self.weight_decay = kwargs.pop("weight_decay", 0.04)
    #     self.clip_value = kwargs.pop("clip_value", 0.0)

    #     self.n_workers = kwargs.pop("n_workers", 0)
    #     self.seed = kwargs.pop("seed", 2020)
    #     self.monitor = kwargs.pop("monitor", True)
    #     self.monitor_only_val = kwargs.pop("monitor_only_val", True)

    #     # self.early_stopping = EarlyStopping(**early_stopping_kwargs)

    #     torch.manual_seed(self.seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(self.seed)
    #         self.model.cuda()
    #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(">> ", self.device)

    #     self.epoch = -1
    #     self.n_epochs = None
    #     self.iter = 0
    #     self.best_epoch = None
    #     self.best_state_dict = None
    #     self.current_loss = None
    #     self.previous_loss_was_nan = False
    #     self.nan_counter = 0
    #     self.optimizer = None
    #     self.training_time = 0

    #     self.train_data = None
    #     self.valid_data = None
    #     self.sampler = None
    #     self.dataloader_train = None
    #     self.dataloader_valid = None

    #     self.iters_per_epoch = None
    #     self.val_iters_per_epoch = None

    #     self.logs = defaultdict(list)

    #     # Create Train/Valid AnnotatetDataset objects
    #     self.train_data, self.valid_data = make_dataset(
    #         self.adata,
    #         train_frac=self.train_frac,
    #         conditions=self.conditions,
    #     )

    # def initialize_loaders(self):
    #     """
    #     Initializes Train-/Test Data and Dataloaders with custom_collate and WeightedRandomSampler for Trainloader.
    #     Returns:
    #     """
    #     if self.n_samples is None or self.n_samples > len(self.train_data):
    #         self.n_samples = len(self.train_data)
    #     self.iters_per_epoch = int(np.ceil(self.n_samples / self.batch_size))

    #     if self.use_stratified_sampling:
    #         # Create Sampler and Dataloaders
    #         stratifier_weights = torch.tensor(self.train_data.stratifier_weights, device=self.device)

    #         self.sampler = WeightedRandomSampler(stratifier_weights,
    #                                              num_samples=self.n_samples,
    #                                              replacement=True)
    #         self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
    #                                                             batch_size=self.batch_size,
    #                                                             sampler=self.sampler,
    #                                                             collate_fn=custom_collate,
    #                                                             num_workers=self.n_workers)
    #     else:
    #         self.dataloader_train = torch.utils.data.DataLoader(dataset=self.train_data,
    #                                                             batch_size=self.batch_size,
    #                                                             shuffle=True,
    #                                                             collate_fn=custom_collate,
    #                                                             num_workers=self.n_workers)
    #     if self.valid_data is not None:
    #         val_batch_size = self.batch_size
    #         if self.batch_size > len(self.valid_data):
    #             val_batch_size = len(self.valid_data)
    #         self.val_iters_per_epoch = int(np.ceil(len(self.valid_data) / self.batch_size))
    #         self.dataloader_valid = torch.utils.data.DataLoader(dataset=self.valid_data,
    #                                                             batch_size=val_batch_size,
    #                                                             shuffle=True,
    #                                                             collate_fn=custom_collate,
    #                                                             num_workers=self.n_workers)

    # # def calc_alpha_coeff(self):
    # #     """Calculates current alpha coefficient for alpha annealing.
    # #        Parameters
    # #        ----------
    # #        Returns
    # #        -------
    # #        Current annealed alpha value
    # #     """
    # #     if self.alpha_epoch_anneal is not None:
    # #         alpha_coeff = min(self.alpha_kl * self.epoch / self.alpha_epoch_anneal, self.alpha_kl)
    # #     elif self.alpha_iter_anneal is not None:
    # #         alpha_coeff = min((self.alpha_kl * (self.epoch * self.iters_per_epoch + self.iter) / self.alpha_iter_anneal), self.alpha_kl)
    # #     else:
    # #         alpha_coeff = self.alpha_kl
    # #     return alpha_coeff

    # def train(self,
    #           n_epochs=400,
    #           lr=1e-3,
    #           eps=0.01):

    #     self.initialize_loaders()
    #     begin = time.time()
    #     self.model.train()
    #     self.n_epochs = n_epochs

    #     params = filter(lambda p: p.requires_grad, self.model.parameters())

    #     self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)

    #     self.before_loop()

    #     for self.epoch in range(n_epochs):
    #         self.on_epoch_begin(lr, eps)
    #         self.iter_logs = defaultdict(list)
    #         for self.iter, batch_data in enumerate(self.dataloader_train):
    #             for key, batch in batch_data.items():
    #                 batch_data[key] = batch.to(self.device)

    #             # Loss Calculation
    #             self.on_iteration(batch_data)

    #         # Validation of Model, Monitoring, Early Stopping
    #         self.on_epoch_end()
    #         if self.use_early_stopping:
    #             if not self.check_early_stop():
    #                 break

    #     # if self.best_state_dict is not None and self.reload_best:
    #     #     print("Saving best state of network...")
    #     #     print("Best State was in Epoch", self.best_epoch)
    #     #     self.model.load_state_dict(self.best_state_dict)

    #     self.model.eval()
    #     self.after_loop()

    #     self.training_time += (time.time() - begin)

    # def before_loop(self):
    #     pass

    # def on_epoch_begin(self, lr, eps):
    #     pass

    # def after_loop(self):
    #     pass

    # def on_iteration(self, batch_data):
    #     # Dont update any weight on first layers except condition weights
    #     # if self.model.freeze:
    #     #     for name, module in self.model.named_modules():
    #     #         if isinstance(module, nn.BatchNorm1d):
    #     #             if not module.weight.requires_grad:
    #     #                 module.affine = False
    #     #                 module.track_running_stats = False

    #     # Calculate Loss depending on Trainer/Model
    #     self.current_loss = loss = self.loss(batch_data)
    #     self.optimizer.zero_grad()
    #     loss.backward()

    #     # Gradient Clipping
    #     if self.clip_value > 0:
    #         torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

    #     self.optimizer.step()

    # def on_epoch_end(self):
    #     # Get Train Epoch Logs
    #     for key in self.iter_logs:
    #         self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())
    #     # Validate Model
    #     if self.valid_data is not None:
    #         self.validate()

    # @torch.no_grad()
    # def validate(self):
    #     self.model.eval()
    #     self.iter_logs = defaultdict(list)
    #     # Calculate Validation Losses
    #     for val_iter, batch_data in enumerate(self.dataloader_valid):
    #         for key, batch in batch_data.items():
    #             batch_data[key] = batch.to(self.device)

    #         val_loss = self.loss(batch_data)

    #     # Get Validation Logs
    #     for key in self.iter_logs:
    #         self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())

    #     self.model.train()

    # def loss(self, total_batch=None):
    #     print("Start test")
    #     print(total_batch["x"].shape)
    #     recon_loss, kl_loss = self.model(**total_batch)
    #     loss = recon_loss + self.calc_alpha_coeff()*kl_loss 
    #     self.iter_logs["loss"].append(loss.item())
    #     self.iter_logs["unweighted_loss"].append(recon_loss.item() + kl_loss.item())
    #     self.iter_logs["recon_loss"].append(recon_loss.item())
    #     self.iter_logs["kl_loss"].append(kl_loss.item())
    #     return loss
