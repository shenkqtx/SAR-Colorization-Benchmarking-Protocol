import numpy as np
from numpy.linalg import norm
import math
import skimage
import torch

def NRMSE_numpy(x_true, x_pred, norm_type = 'euclidean'):
    '''
    :param x_true: target image, shape like [H, W, C]
    :param x_pred: predict image, shape like [H, W, C]
    :param norm_type: {‘euclidean’, ‘min-max’, ‘mean’}
    :return: normalized rmse value
    '''
    m, n, c = x_true.shape
    scores = []
    for i in range(c):
        if norm_type == 'euclidean':
            denorm = np.sqrt(np.mean(x_true[:,:,i] * x_true[:,:,i]))
        elif norm_type == 'min-max':
            denorm = x_true[:,:,i].max() - x_true[:,:,i].min()
        else:
            denorm = x_true[:,:,i].mean()

        scores.append(np.sqrt(np.mean((x_true[:, :, i] - x_pred[:, :, i]) ** 2)) / denorm)

    return np.mean(scores)


def SAM_numpy(x_true, x_pred):
    """
    :param x_true：格式：(H, W, C)
    :param x_pred：格式：(H, W, C)
    :return: angle °
    """
    M, N = x_true.shape[0], x_true.shape[1]

    prod_scal = np.sum(x_true * x_pred, axis=2)
    norm_orig, norm_fusa = np.sum(x_true * x_true, axis=2), np.sum(x_pred * x_pred, axis=2)
    prod_norm = np.sqrt(norm_orig * norm_fusa)

    prod_scal, prod_norm = prod_scal.reshape(M * N, 1), prod_norm.reshape(M * N, 1)

    z = np.where(prod_norm != 0)
    prod_norm, prod_scal = prod_norm[z], prod_scal[z]

    res = np.clip(prod_scal / prod_norm, -1, 1)
    res = np.arccos(res)
    sam = np.mean(res)
    sam = sam * 180 / math.pi
    return sam


def SAM_torch(x_true, x_pred):

    dot_sum = torch.sum(x_true * x_pred, dim=1)
    norm_true = torch.norm(x_true, dim=1)
    norm_pred = torch.norm(x_pred, dim=1)

    res = dot_sum / (norm_pred * norm_true)
    res = torch.clamp(res,-1,1)
    res = torch.acos(res) * 180 / math.pi   # degree
    sam = torch.mean(res)
    return sam