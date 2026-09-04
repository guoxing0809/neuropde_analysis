import os
import yaml
import torch
import numpy as np
import cupy as cp
import scipy.stats as stats
from scipy.interpolate import griddata, RBFInterpolator, CloughTocher2DInterpolator
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse import spmatrix as cpu_spmatrix
from cupyx.scipy.sparse import spmatrix as gpu_spmatrix



def load_yaml_params(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    return configs


def tensor2device(data, device):
    """
    Recursively move torch tensors to specified device (CPU/CUDA)
    support data type: dict, list, tuple, torch.Tensor
    """
    if isinstance(data, dict):
        return {k: tensor2device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(tensor2device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def array2gpu(data):
    """
    Recursively move data to GPU (CuPy/CuPy Sparse)
    support data type：dict, list, tuple, numpy array, scipy sparse
    """
    if isinstance(data, dict):
        return {k: array2gpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(array2gpu(v) for v in data)
    elif isinstance(data, (np.ndarray, cpu_spmatrix)):
        # sparse
        if isinstance(data, cpu_spmatrix):
            if data.format == 'csr': return cp.sparse.csr_matrix(data)
            if data.format == 'coo': return cp.sparse.coo_matrix(data)
            if data.format == 'csc': return cp.sparse.csc_matrix(data)
            return cp.sparse.csr_matrix(data)
        # dense
        return cp.asarray(data)
    else:
        return data

def array2cpu(data):
    """
    Recursively move data to CPU (NumPy/Scipy Sparse)
    support data type：dict, list, tuple, cupy array, cupyx sparse
    """
    if isinstance(data, dict):
        return {k: array2cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(array2cpu(v) for v in data)
    elif hasattr(data, 'get'):
        # CuPy objects (dense and sparse) use .get() to return to CPU
        return data.get()
    else:
        return data


def inf_norm_error(u, u_ref, normalize=True):
    abs_err = np.abs(u - u_ref)
    inf_norm_err = np.max(abs_err, axis=0)
    if normalize:
        scale = np.max(np.abs(u_ref), axis=0)
        inf_norm_err /= scale
    return inf_norm_err


def l2_norm_error(u, u_ref):
    abs_err = np.abs(u - u_ref)
    l2_norm_err = np.linalg.norm(abs_err, axis=0) / np.linalg.norm(u_ref, axis=0)
    return l2_norm_err


def spearman_coef(u, u_ref):
    if u_ref.ndim > 1:
        coef = []
        for i in range(u_ref.shape[1]):
            coef_i = stats.spearmanr(u[:, i], u_ref[:, i]).correlation
            coef.append(coef_i)
    else:
        coef = stats.spearmanr(u, u_ref).correlation
    return coef


def pearson_coef(u, u_ref):
    if u_ref.ndim > 1:
        coef = []
        for i in range(u_ref.shape[1]):
            coef_i = stats.pearsonr(u[:, i], u_ref[:, i])[0]
            coef.append(coef_i)
    else:
        coef = stats.pearsonr(u, u_ref)[0]
    return coef


def griddata_interpolate(base_points, base_values, int_points, method='linear'):
    # 2D interpolate
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    if base_points.shape[0] != base_values.shape[0]:
        raise ValueError(f'The number of base points mismatch the values!')
    int_values = np.zeros((int_points.shape[0], base_values.shape[1]), dtype=np.float64)
    for i in range(base_values.shape[1]):
        int_values[:, i] = griddata(base_points, base_values[:, i], int_points, method, fill_value=0)
    return int_values

def rbf_interpolate(base_points, base_values, int_points, method='linear', neighbors=64):
    # 2D interpolate
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator
    if base_points.shape[0] != base_values.shape[0]:
        raise ValueError(f'The number of base points mismatch the values!')
    base_points = np.ascontiguousarray(base_points, dtype=np.float64)
    base_values = np.ascontiguousarray(base_values, dtype=np.float64)
    int_points = np.ascontiguousarray(int_points,  dtype=np.float64)
    int_values = np.zeros((int_points.shape[0], base_values.shape[1]), dtype=np.float64)
    for i in range(base_values.shape[1]):
        interp = RBFInterpolator(base_points, base_values[:, i], kernel=method, neighbors=neighbors)
        int_values[:, i] = interp(int_points)
    return int_values

def cloughtoucher_interpolate(base_points, base_values, int_points):
    # 2D interpolate
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html#scipy.interpolate.CloughTocher2DInterpolator
    if base_points.shape[0] != base_values.shape[0]:
        raise ValueError(f'The number of base points mismatch the values!')
    int_values = np.zeros((int_points.shape[0], base_values.shape[1]), dtype=np.float64)
    for i in range(base_values.shape[1]):
        interp = CloughTocher2DInterpolator(base_points, base_values[:, i], fill_value=0.0)
        int_values[:, i] = interp(int_points)
    return int_values

def idw_interpolate(base_points, base_values, int_points, n_neighbors=3, weights='distance', algorithm='auto'):
    # 2D interpolate
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    if base_points.shape[0] != base_values.shape[0]:
        raise ValueError(f'The number of base points mismatch the values!')
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    if base_values.ndim == 1:
        knn.fit(base_points, base_values)
        int_values = knn.predict(int_points)
    else:
        int_values = np.zeros((int_points.shape[0], base_values.shape[1]), dtype=np.float64)
        for i in range(base_values.shape[1]):
            knn.fit(base_points, base_values[:, i])
            int_values[:, i] = knn.predict(int_points)
    return int_values



