import torch
import numpy as np
import scanpy as sc
from scipy import sparse
import collections.abc as container_abcs
import re

def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot

def remove_sparsity(adata):
    """
        If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
    """
    if sparse.issparse(adata.X):
        new_adata = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        return new_adata

    return adata

def compute_out_size(in_size, kernel_size, stride, padding, dilation=1):
    """
    Compute the images sizes after applying a convolution.

    See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    return int(np.floor(((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1))




def label_encoder(adata, encoder, condition_key=None):
    """Encode labels of Annotated `adata` matrix.
       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix.
       encoder: Dict
            dictionary of encoded labels.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       Returns
       -------
       labels: `~numpy.ndarray`
            Array of encoded labels
       label_encoder: Dict
            dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(encoder.keys())):
        print(f"Warning: Labels in adata.obs[{condition_key}] is not a subset of label-encoder!")
        print("Therefore integer value of those labels is set to -1")
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    return labels


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, container_abcs.Mapping):
        output = {key: custom_collate([d[key] for d in batch]) for key in elem}
        return output