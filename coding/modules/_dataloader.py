from .annotated_dataset import AnnotatedDataset
import numpy as np
from sklearn.model_selection import train_test_split as tts
import scanpy as sc
import anndata as ad
import pytorch_lightning as pl

from scipy import sparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

# def make_dataset(adata,
#                  train_frac=0.9,
#                  conditions=None,
#                  ):
#     """Splits 'adata' into train and validation data and converts them into 'CustomDatasetFromAdata' objects.
#        Parameters
#        ----------
#        Returns
#        -------
#        Training 'CustomDatasetFromAdata' object, Validation 'CustomDatasetFromAdata' object
#     # """
#     # size_factors = adata.X.sum(1)
#     # if len(size_factors.shape) < 2:
#     #     size_factors = np.expand_dims(size_factors, axis=1)
#     # adata.obs['trvae_size_factors'] = size_factors

#     # Preprare data for semisupervised learning
#     labeled_array = np.zeros((len(adata), 1))
#     # if labeled_indices is not None:
#     #     labeled_array[labeled_indices] = 1
#     # adata.obs['trvae_labeled'] = labeled_array

#     # if cell_type_keys is not None:
#     #     finest_level = None
#     #     n_cts = 0
#     #     for cell_type_key in cell_type_keys:
#     #         if len(adata.obs[cell_type_key].unique().tolist()) >= n_cts:
#     #             n_cts = len(adata.obs[cell_type_key].unique().tolist())
#     #             finest_level = cell_type_key

#         # train_adata, validation_adata = train_test_split(adata, train_frac, cell_type_key=finest_level)

#     # if conditions is not None:
#     #     train_adata, validation_adata = train_test_split(adata, train_frac, condition_key=condition_key)
#     # else:
#     train_adata, validation_adata = train_test_split(adata, train_frac)
#     return train_adata, validation_adata
#     # data_set_train = AnnotatedDataset(
#     #     train_adata,
#     #     condition_key=condition_key,
#     #     cell_type_keys=cell_type_keys,
#     #     condition_encoder=condition_encoder,
#     #     cell_type_encoder=cell_type_encoder,
#     # )
#     # if train_frac == 1:
#     #     return data_set_train, None
#     # else:
#     #     data_set_valid = AnnotatedDataset(
#     #         validation_adata,
#     #         condition_key=condition_key,
#     #         cell_type_keys=cell_type_keys,
#     #         condition_encoder=condition_encoder,
#     #         cell_type_encoder=cell_type_encoder,
#     #     )
#     #     return data_set_train, data_set_valid


class CVAEDataModule(pl.LightningDataModule):
    """Data module."""

    def __init__(
        self,
        adata,
        dataset_args: dict,
        test_split_seed: int = 0,
        val_split_seed: int = 0,
        test_size: float = 0.1,
        val_size: float = 0.1,
        batch_size: int = 64,
        scale : bool = False,
    ):
        super(CVAEDataModule, self).__init__()
        self.adata = adata

        self.conditions = dataset_args["conditions"]
        self.dataset_name = dataset_args["dataset_name"]

        self.batch_size = batch_size
        self.val_split_seed = val_split_seed
        self.test_split_seed = test_split_seed
        self.test_size = test_size
        self.val_size = val_size
        self.scale = scale

    def prepare_data(
        self,
        is_preprocessed: bool = True,
        filter_min_counts: bool = True,
        log_trans_input: bool = True,
    ):
        if is_preprocessed:
            pass
        else:
            # prepare RNA data
            # if filter_min_counts:
            #     sc.pp.filter_genes(self.adata, min_counts=1)
            #     sc.pp.filter_cells(self.adata, min_counts=1)

            if sparse.issparse(self.adata.X):
                self.adata.X = self.adata.X.A

            adata_count = self.adata.copy()
            obs = self.adata.obs_keys()

            # if size_factors and self.size_factors_key not in obs:
            #     sc.pp.normalize_total(
            #         self.adata, target_sum=target_sum, exclude_highly_expressed=False, key_added=self.size_factors_key
            #     )

            # if log_trans_input:
            #     sc.pp.log1p(self.adata)
            #     if 0 < n_top_genes < self.adata.shape[1]:
            #         sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
            #         genes = self.adata.var["highly_variable"]
            #         self.adata = self.adata[:, genes]
            #         adata_count = adata_count[:, genes]

            if self.scale:
                sc.pp.scale(self.adata)

            if self.adata.raw is None:
                self.adata.raw = adata_count.copy()

            # prepare image data
            # read the image data stored as tif file

            # if self.dataset_name not in self.adata.uns:
            #     self.adata.uns[self.dataset_name] = {}
            # self.adata.uns[self.dataset_name][self.image_key] = img

    def setup(self):
        # perform splits
        indices = np.arange(self.adata.shape[0])

        train_val_indices, test_indices = tts(
            indices, test_size=self.test_size, shuffle=True, random_state=self.test_split_seed
        )

        train_indices, val_indices = tts(
            train_val_indices, test_size=self.val_size, shuffle=True, random_state=self.val_split_seed
        )

        self.train = self.adata[train_indices, :]
        self.val = self.adata[val_indices, :]
        self.test = self.adata[test_indices, :]

        self.train = DatasetFromAdata(
            self.train,
            self.dataset_name,
            self.conditions,
        )
        self.val = DatasetFromAdata(
            self.val,
            self.dataset_name,
            self.conditions,
        )
        self.test = DatasetFromAdata(
            self.test,
            self.dataset_name,
            self.conditions,
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def latent_dataloader(self):
        ds_full = DatasetFromAdata(
            self.adata,
            self.dataset_name,
            self.spatial_coord_key,
            self.size_factors_key,
            self.image_key,
            self.image_radius,
            self.resize,
        )
        return DataLoader(ds_full, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
    def train_test_split(adata, train_frac=0.85):
        """Splits 'Anndata' object into training and validation data.
          Parameters
          ----------
          adata: `~anndata.AnnData`
                `AnnData` object for training the model.
          train_frac: float
                Train-test split fraction. the model will be trained with train_frac for training
                and 1-train_frac for validation.
          Returns
          -------
          `AnnData` objects for training and validating the model.
        """
        if train_frac == 1:
            return adata, None
        else:
            indices = np.arange(adata.shape[0])

        n_train_samples = int(np.ceil(train_frac * len(indices)))
        np.random.shuffle(indices)
        train_idx = indices[:n_train_samples]
        val_idx = indices[n_train_samples:]

        train_data = adata[train_idx, :]
        valid_data = adata[val_idx, :]

        return train_data, valid_data

class DatasetFromAdata(Dataset):
    """Custom dataset class."""

    def __init__(
        self,
        adata: ad.AnnData,
        dataset_name: str,
        condition_names: list,
    ):

        self.adata = adata
        self.dataset_name = dataset_name

        if sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.A

        self.data = self.adata.X
        self.conditions = self.adata.obs[condition_names].values
        self.conditions_scaled = self.coordinates_ptp(self.conditions)

    def __getitem__(self, index):
        items = {}

        items["rna"] = torch.tensor(self.data[index, :], requires_grad=True)
        items["coord"] = torch.tensor(self.conditions_scaled[index, :], dtype=torch.float32, requires_grad=True).float()

        return items

    def coordinates_ptp(coords, eps=1e-8):
      """Normalize the coordinates."""
      return (coords - np.min(coords, axis=0)) / (np.ptp(coords, axis=0) + eps)

    def __len__(self):
        return len(self.adata)



