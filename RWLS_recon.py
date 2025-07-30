import torch
import matplotlib.pyplot as plt
import leapctype as leap
import leaptorch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import math
import glob
import imageio.v2 as imageio
import tifffile
from tqdm.notebook import tqdm
import bm3d

def key_func(fname):
    return int(fname.split('theta')[-1].split('deg')[0])

fnames = sorted(
    [f for f in glob.glob('/egr/research-slim/shared/Phase1Datasets/LowSNR/*.tif')
     if 'FullResponse' in f],
    key=key_func
)

# torch.tensor will copy and up‐cast uint16 → the default dtype (float32)
rad = torch.stack([
    torch.from_numpy(imageio.imread(f).astype(np.float32))
    for f in fnames
]).unsqueeze(0)
rad = torch.cat([chunk.mean(1, keepdim=True) for chunk in torch.chunk(rad, 20, dim=1)], dim=1)
rad = torch.cat([torch.from_numpy(bm3d.bm3d(rad[0,i], rad[0,i].std())).unsqueeze(0).unsqueeze(0) for i in tqdm(range(rad.shape[1]))], dim=1).float()
print(rad.shape)
proj = -(rad/rad.max()).log().float()#-((rad+1)/2**16).log()
angles = torch.from_numpy(np.array([key_func(fname) for fname in fnames])).float()

angles = torch.unique(angles)

class PBCT:

    def __init__(self, num_views, num_rows, num_cols, device='cpu', angles=None):

        pixelHeight = 1
        pixelWidth = 1
        centerRow = num_rows//2
        centerCol = num_rows//2
        device = torch.device(device)

        if device == torch.device('cpu'):
            self.proj = leaptorch.Projector(forward_project=True, use_gpu=False, gpu_device=device, batch_size=1)
        else:
            self.proj = leaptorch.Projector(forward_project=True, use_gpu=True, gpu_device=device, batch_size=1)
        
        if angles is None:
            phis = self.proj.leapct.setAngleArray(num_views, 180.0)
        else:
            phis = angles

        self.proj.leapct.set_parallelbeam(num_views, num_rows, num_cols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

        self.proj.leapct.set_diameterFOV(1.0e7)

        self.proj.set_default_volume()
        self.proj.allocate_batch_data()
        self.proj.leapct.set_truncatedScan(True)

    def A(self, x):
        x = x.float().contiguous()
        return self.proj(x).clone()

    def A_T(self, y):
        y = y.float().contiguous()
        self.proj.forward_project = False
        x = self.proj(y)
        self.proj.forward_project = True
        return x.clone()

    def A_pinv(self, y):
        y = y.float().contiguous()
        return self.proj.fbp(y)
    
ct = PBCT(20, 400, 400, angles=angles, device='cpu')

proj = torch.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)

fbp = ct.A_pinv(proj)
bp = ct.A_T(proj)

fbp = ct.A_pinv(proj)



filters = leap.filterSequence(1.0)
filters.append(leap.TV(ct.proj.leapct, weight=1.0e3, delta=1e-4))
tv_recon = ct.proj.leapct.RWLS(proj.squeeze(), torch.zeros_like(fbp).squeeze(), 200, filters=filters, nonnegativityConstraint=True)

np.save(os.path("RWLS_Recon.npy",tv_recon))