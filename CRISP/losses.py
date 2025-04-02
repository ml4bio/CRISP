"""
This module contains various loss functions used for training CRISP models.
It includes specialized losses for drug perturbation prediction tasks, such as
adaptive losses, MMD-based losses, and losses focused on differentially expressed genes.
"""

import torch
import torch.nn.functional as F
from torch import nn
from geomloss import SamplesLoss

def loss_adapt(pred,true,mean_ctrl,std_ctrl,thres=2,std_thres=0.01):
    """
    Calculates an adaptive loss that normalizes predictions and ground truth by control statistics
    and focuses on genes with significant expression changes.
    
    Args:
        pred: Predicted gene expression values
        true: True gene expression values
        mean_ctrl: Mean expression values in control state
        std_ctrl: Standard deviation of expression values in control state
        thres: Threshold to determine significant expression changes (default: 2)
        std_thres: Minimum standard deviation threshold (default: 0.01)
        
    Returns:
        Normalized mean squared error between significant predicted and true expression changes
    """
    std_ctrl += 1e-8
    std_ctrl = std_ctrl.clamp(min=std_thres)
    pred_delta = ((pred-mean_ctrl)/std_ctrl)
    true_delta = ((true-mean_ctrl)/std_ctrl)

    mask = torch.logical_or((pred_delta**2)>(thres**2),(true_delta**2)>(thres**2))
    # mask = torch.logical_and(mask_p,std_ctrl>std_thres)

    return torch.sum(((pred_delta-true_delta).clamp(min=-10,max=10)**2) * mask) / pred.shape[0] / pred.shape[1]


class RBF(nn.Module):
    """
    Radial Basis Function kernel implementation for MMD calculations.
    Uses multiple bandwidth values for better representation.
    
    Args:
        n_kernels: Number of kernels with different bandwidths (default: 5)
        mul_factor: Multiplication factor between consecutive bandwidth values (default: 2.0)
        bandwidth: Fixed bandwidth value, if None it's estimated from data (default: None)
        device: Computation device (default: 'cuda')
    """

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None,device='cuda'):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """
        Determines the bandwidth parameter for the RBF kernel.
        
        Args:
            L2_distances: Pairwise squared Euclidean distances
            
        Returns:
            Bandwidth value (either predefined or estimated from data)
        """
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        """
        Computes the RBF kernel matrix for the input data.
        
        Args:
            X: Input data tensor
            
        Returns:
            Kernel matrix with summed contributions from multiple bandwidths
        """
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

def MMDloss(X,Y,device='cuda'):
    """
    Computes Maximum Mean Discrepancy loss between two distributions.
    
    Args:
        X: Samples from first distribution
        Y: Samples from second distribution
        device: Computation device (default: 'cuda')
        
    Returns:
        MMD distance between distributions X and Y
    """

    kernel = RBF(device=device)
    K = kernel(torch.vstack([X, Y]))
    X_size = X.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()

    return XX - 2 * XY + YY

# def MMD_group(group_idx,true, pred,device='cuda'):
#     """
#     Computes group-wise MMD loss between true and predicted values.
#     
#     Args:
#         group_idx: Group indices for samples
#         true: True values
#         pred: Predicted values
#         device: Computation device (default: 'cuda')
#         
#     Returns:
#         Average MMD loss across groups
#     """
#     l_ = torch.tensor([0.0],device=device)
#     for i in set(group_idx):
#         mask = group_idx==i.item()
#         # l = MMDloss(true[mask,:],pred[mask,:])
#         l = compute_mmd(true[mask,:],pred[mask,:])
#         l_ += l

#     return l_/len(set(group_idx))

def gaussian_mmd(x,y,blur=1):
    """
    Computes Gaussian MMD loss using GeomLoss library.
    
    Args:
        x: Samples from first distribution
        y: Samples from second distribution
        blur: Bandwidth parameter (default: 1)
        
    Returns:
        Gaussian MMD distance
    """

    loss_f = SamplesLoss(loss='gaussian',blur=blur).to(x.device)
    return loss_f(x,y)

def sinkhorn_dist(x,y,blur=.05):
    """
    Computes Sinkhorn distance (entropy-regularized optimal transport).
    
    Args:
        x: Samples from first distribution
        y: Samples from second distribution
        blur: Regularization parameter (default: 0.05)
        
    Returns:
        Sinkhorn distance
    """
    sink = SamplesLoss(loss="sinkhorn",blur=blur).to(x.device)
    return sink(x,y)

def energy_dist(x,y):
    """
    Computes energy distance between distributions.
    
    Args:
        x: Samples from first distribution
        y: Samples from second distribution
        
    Returns:
        Energy distance
    """
    Edist = SamplesLoss(loss='energy').to(x.device)
    return Edist(x,y)

class AFMSELoss(torch.nn.Module):
    """
    MSE loss focused specifically on differentially expressed genes (DEGs).
    Only computes loss on genes marked as differentially expressed.
    
    This loss helps the model focus on accurately predicting genes that change
    significantly in response to perturbations.
    """
    def __init__(self):
        super().__init__()
    def forward(self,y,pred,degs):
        """
        Computes MSE loss on differentially expressed genes.
        
        Args:
            y: True gene expression values
            pred: Predicted gene expression values
            degs: Mask of differentially expressed genes (1 for DE, 0 otherwise)
            
        Returns:
            Mean squared error on differentially expressed genes
        """
        degs = degs.float()
        y_de = y * degs
        pred_de = pred * degs
        mse = (y_de - pred_de)**2
        num_degs = degs.sum(axis=1)
        mse = mse.sum(axis=1)/(num_degs+1e-6)
        mse = sum(mse)/(len(torch.nonzero(num_degs)) + 1e-6)
        return mse
    
def HVGPRLoss(true,pred,hvgs):
    """
    Computes loss based on Pearson correlation between true and predicted values
    for highly variable genes (HVGs).
    
    Args:
        true: True gene expression values
        pred: Predicted gene expression values
        hvgs: Mask of highly variable genes (1 for HVG, 0 otherwise)
        
    Returns:
        1 minus the mean Pearson correlation (lower is better)
    """
    hvgs = hvgs.float()
    y_hvg = true * hvgs
    pred_hvg = pred * hvgs

    cov_xy = (y_hvg*pred_hvg).mean(1)-y_hvg.mean(1)*pred_hvg.mean(1)
    std_x = torch.sqrt(((y_hvg-y_hvg.mean(1).unsqueeze(1))**2).mean(1))
    std_y = torch.sqrt(((pred_hvg-pred_hvg.mean(1).unsqueeze(1))**2).mean(1))      
    pr = cov_xy / ((std_x * std_y) + 1e-8)

    return 1-pr.mean()



class AFMSELoss_wei(torch.nn.Module):
    """
    Weighted MSE loss on differentially expressed genes.
    Weights the MSE based on the variance of gene expression,
    giving more importance to genes with stable expression patterns.
    """
    def __init__(self):
        super().__init__()
    def forward(self,y,pred,degs):
        """
        Computes weighted MSE loss on differentially expressed genes.
        
        Args:
            y: True gene expression values
            pred: Predicted gene expression values
            degs: Mask of differentially expressed genes (1 for DE, 0 otherwise)
            
        Returns:
            Weighted mean squared error on differentially expressed genes
        """
        y_de = y * degs
        pred_de = pred * degs
        mse = (y_de - pred_de)**2
        num_degs = degs.sum(axis=1)
        mse = mse.sum(axis=1)/(num_degs+1e-6)

        treat = torch.nonzero(num_degs)
        degs = degs.bool()
        selected_elements = y[degs]
        split_tensor = selected_elements.reshape((treat.shape[0],50))
        variances_per_row = split_tensor.var(dim=1)
        weight_clip = (1/variances_per_row).clip(0.3,2)
        weight = torch.zeros((y.shape[0]),device=mse.device)
        weight[treat[:,0]] = weight_clip

        loss = sum(mse * weight)/treat.shape[0]

        return loss
    
def dir_loss(y, pred, degs, ctrl):
    """
    Direction loss focusing on correctly predicting the direction of expression changes
    relative to the control state.
    
    Args:
        y: True gene expression values
        pred: Predicted gene expression values
        degs: Mask of differentially expressed genes
        ctrl: Control state gene expression values
        
    Returns:
        Mean squared error of the sign differences between true and predicted expression changes
    """
    degs = degs.float()
    y_de = y * degs
    pred_de = pred * degs
    ctrl_de = ctrl * degs
    dir = (F.softsign(y_de - ctrl_de) - F.softsign(pred_de - ctrl_de))**2
    num_degs = degs.sum(axis=1)
    dir = dir.sum(axis=1)/(num_degs+1e-6)
    dir = sum(dir)/len(torch.nonzero(num_degs)+1e-6)

    return dir