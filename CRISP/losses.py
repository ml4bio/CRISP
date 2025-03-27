
import torch
import torch.nn.functional as F
from torch import nn
from geomloss import SamplesLoss

def loss_adapt(pred,true,mean_ctrl,std_ctrl,thres=2,std_thres=0.01):

    std_ctrl += 1e-8
    std_ctrl = std_ctrl.clamp(min=std_thres)
    pred_delta = ((pred-mean_ctrl)/std_ctrl)
    true_delta = ((true-mean_ctrl)/std_ctrl)

    mask = torch.logical_or((pred_delta**2)>(thres**2),(true_delta**2)>(thres**2))
    # mask = torch.logical_and(mask_p,std_ctrl>std_thres)

    return torch.sum(((pred_delta-true_delta).clamp(min=-10,max=10)**2) * mask) / pred.shape[0] / pred.shape[1]


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None,device='cuda'):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

def MMDloss(X,Y,device='cuda'):

    kernel = RBF(device=device)
    K = kernel(torch.vstack([X, Y]))
    X_size = X.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()

    return XX - 2 * XY + YY

# def MMD_group(group_idx,true, pred,device='cuda'):
#     l_ = torch.tensor([0.0],device=device)
#     for i in set(group_idx):
#         mask = group_idx==i.item()
#         # l = MMDloss(true[mask,:],pred[mask,:])
#         l = compute_mmd(true[mask,:],pred[mask,:])
#         l_ += l

#     return l_/len(set(group_idx))

def gaussian_mmd(x,y,blur=1):
    loss_f = SamplesLoss(loss='gaussian',blur=blur).to(x.device)
    return loss_f(x,y)

def sinkhorn_dist(x,y,blur=.05):
    sink = SamplesLoss(loss="sinkhorn",blur=blur).to(x.device)
    return sink(x,y)

def energy_dist(x,y):
    Edist = SamplesLoss(loss='energy').to(x.device)
    return Edist(x,y)

class AFMSELoss(torch.nn.Module):
    """ 
    MSE loss on DE genes,
    take true, prediction and de genes mask as input
    """
    def __init__(self):
        super().__init__()
    def forward(self,y,pred,degs):
        degs = degs.float()
        y_de = y * degs
        pred_de = pred * degs
        mse = (y_de - pred_de)**2
        num_degs = degs.sum(axis=1)
        mse = mse.sum(axis=1)/(num_degs+1e-6)
        mse = sum(mse)/(len(torch.nonzero(num_degs)) + 1e-6)
        return mse
    
def HVGPRLoss(true,pred,hvgs):
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
    Weighted MSE loss on DE genes, remains to be investigated
    take true, prediction and de genes mask as input
    """
    def __init__(self):
        super().__init__()
    def forward(self,y,pred,degs):
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
    Direction loss on DE genes regarding to control state, measuring error on predicted response direction
    could be seen as addition to MSE loss
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