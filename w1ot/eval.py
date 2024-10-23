import os
import anndata
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from w1ot.utils import ensure_numpy
from sklearn.metrics.pairwise import rbf_kernel


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def metrics(transported, target, gene_list=None,data_space='X'):
    if data_space == 'X':
        if gene_list is not None:
            transported_data = ensure_numpy(transported[:, gene_list].X)
            target_data = ensure_numpy(target[:, gene_list].X)
        else:
            transported_data = ensure_numpy(transported.X)
            target_data = ensure_numpy(target.X)

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
    gammas = np.logspace(1, -3, num=50)
    mmd_dist = np.mean([mmd_distance(target_data, transported_data, g) for g in gammas])

    return r2, l2_dist, mmd_dist


