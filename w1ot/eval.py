import os
import anndata
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import rbf_kernel

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))

def evaluate(transported, target, data_space='X'):
    if data_space == 'X':
        transported_data = transported.X
        target_data = target.X
    elif 'emb' in data_space:
        transported_data = transported.obsm[data_space]
        target_data = target.obsm[data_space]
    else:
        raise ValueError(f"Unknown data_space: {data_space}")

    # Compare the feature means
    transported_mean = transported_data.mean(axis=0).flatten()
    target_mean = target_data.mean(axis=0).flatten()

    # Pearson correlation coefficient between means
    corr_coeff, _ = pearsonr(transported_mean, target_mean)
    r2 = corr_coeff ** 2

    # L2 distance between means
    l2_dist = np.linalg.norm(transported_mean - target_mean)

    # MMD distance
    mmd_dist = compute_scalar_mmd(target_data, transported_data)

    return r2, l2_dist, mmd_dist


