import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
import deepinv as dinv
from skimage.transform import iradon
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from itertools import repeat
import odl
from odl.phantom import ellipsoid_phantom
from odl import uniform_discr
from Model_arch import UNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import TVPrior
from deepinv.optim.optimizers import BaseOptim, create_iterator
from skimage.metrics import structural_similarity as _ssim_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SSIMBaseOptim(BaseOptim):
    """
    A BaseOptim that computes SSIM in addition to PSNR, residual, (cost).
    """

    def init_metrics_fn(self, X_init, x_gt=None):
        
        metrics = super().init_metrics_fn(X_init, x_gt=x_gt)
        
        metrics["ssim"] = [[] for _ in range(self.batch_size)]
        return metrics

    def update_metrics_fn(self, metrics, X_prev, X, x_gt=None):
        
        metrics = super().update_metrics_fn(metrics, X_prev, X, x_gt=x_gt)

        # Now append SSIM for each item in batch
        if x_gt is not None:
            x   = self.get_output(X)    # current estimate tensor [B,C,H,W]
            gt  = x_gt                  # ground truth tensor [B,C,H,W]
            x_np  = x.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()

            for i in range(self.batch_size):
                # compute data_range from ground truth
                data_range = float(gt_np[i].max() - gt_np[i].min())
                # assume single-channel C=1
                val = _ssim_fn(gt_np[i,0], x_np[i,0], data_range=data_range)
                metrics["ssim"][i].append(val)

        return metrics
    
angles_torch = torch.linspace(0, 180, 60, device=device)

physics = dinv.physics.Tomography(
        img_width=256, 
        angles=angles_torch, 
        device=device,
        )
    
def classic_TV_solver(lambs, image_path, noise_level = "none"):
    
    walnut_data_noisy = torch.load(image_path, map_location=device)
    walnut_data = walnut_data_noisy.to(device)

    walnut_GT = torch.load("walnut.pt", map_location=device)

    # w_mean = walnut_data.mean()
    # w_std = walnut_data.std()
    # walnut_data = (walnut_data - w_mean) / (w_std + 1e-10)  # Standardise the input data
    # walnut_data = torch.clamp(walnut_data, 0, 1) # Ensure values are in [0, 1]

    A_norm = physics.compute_norm(physics.A_dagger(walnut_data)).item()
    print(f"Operator norm A: {A_norm:.2f}")
    stepsize = 2 / (A_norm + 1e-12)  
    print(f"Computed stepsize: {stepsize:.1e}") 
    
    data_fidelity = L2()
    prior = TVPrior(n_it_max=20)

    # Algorithm parameters
    verbose = True
    max_iter = 6000
    early_stop = True

    best_psnr = -float('inf')
    best_lamb = None
    best_stepsize = None
    best_ssim = -float('inf')
    psnr_curves = {}
    ssim_curves = {}
    for lamb in lambs:

        print(f"Running TV-PGD with lambda {lamb} for noise level {noise_level}...")

        params_algo = {"stepsize": stepsize, "lambda": lamb}
        iterator = create_iterator("PGD", prior=prior)

        model = SSIMBaseOptim(
            iterator         = iterator,
            params_algo      = params_algo,
            data_fidelity    = data_fidelity,
            prior            = prior,
            max_iter         = max_iter,
            early_stop       = early_stop,
            verbose          = verbose,
            has_cost         = True,    # so BaseOptim computes cost & PSNR
        )

        x_model, metrics = model(
                y = walnut_data,
                physics = physics,
                x_gt = walnut_GT,
                compute_metrics=True,
            )

        rec_np = x_model.squeeze().detach().cpu().numpy()
        plt.imshow(rec_np, cmap='gray', vmin=0, vmax=1)
        plt.title(f"TV‐PGD λ={lamb:.1e}, PSNR={metrics['psnr'][-1][-1]:.1f} dB, SSIM={metrics['ssim'][-1][-1]:.1f} for noise level {noise_level}")
        plt.axis('off')
        os.makedirs(f"results/classic/tv_sigma_{noise_level}", exist_ok=True)
        plt.savefig(f"results/classic/tv_sigma_{noise_level}/rec_lambda_{lamb:.0e}_step_size{stepsize:.0e}.png", dpi=200)
        plt.close()

        print(f"first PSNR {metrics['psnr'][-1][0]}")
        print(f"first SSIM {metrics['ssim'][-1][0]}")
        psnr_curve = metrics['psnr'][-1] # Get the PSNR curve from the metrics
            # store the full PSNR-vs-iteration curve for this lambda
        psnr_curves[lamb] = psnr_curve
        ssim_curve = metrics['ssim'][-1] # Get the SSIM curve from the metrics
        ssim_curves[lamb] = ssim_curve
        


        final_psnr = psnr_curve[-1]
        final_ssim = ssim_curve[-1]
        print(f"For λ={lamb:.1e}: Final PSNR  {final_psnr:.1f} dB, Final SSIM = {final_ssim:.1f} for noise level {noise_level}")

        if final_psnr > best_psnr and final_ssim > best_ssim:
                best_psnr = final_psnr
                best_ssim = final_ssim
                best_lamb = lamb
                best_stepsize = stepsize
                print(f"New best: PSNR={best_psnr:.2f} dB, SSIM={best_ssim:.3f} (lambda ={best_lamb:.2f} for noise level {noise_level})")

    # 2) Plot PSNR‐vs‐iterations for each λ
    plt.figure(figsize=(6,4))
    for lamb, curve in psnr_curves.items():
        plt.plot(curve, label=f"lambda={lamb:.0e}")

    plt.xlabel("PGD iteration")
    plt.ylabel("PSNR [dB]")
    title = (
    f"PSNR trajectories\n"
    f"best λ={best_lamb:.0e}, step={best_stepsize:.0e}, "
    f"PSNR={best_psnr:.2f} dB (noise={noise_level})"
    )
    plt.title(title, fontsize=8)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"results/classic/tv_sigma_{noise_level}/psnr_trajectories_{best_lamb}_{best_stepsize}_{noise_level}.png", dpi=200)
    plt.close()

    # 3) Plot SSIM‐vs‐iterations for each λ
    plt.figure(figsize=(6,4))
    for lamb, curve in ssim_curves.items():
        plt.plot(curve, label=f"lambda={lamb:.0e}")
    plt.xlabel("PGD iteration")
    plt.ylabel("SSIM")
    title = (
        f"SSIM trajectories\n"
        f"best λ={best_lamb:.0e}, step={best_stepsize:.0e}, "
        f"SSIM={best_ssim:.2f} (noise={noise_level})"
    )
    plt.title(title, fontsize=8)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"results/classic/tv_sigma_{noise_level}/ssim_trajectories_{best_lamb}_{best_stepsize}_{noise_level}.png", dpi=200)
    plt.close()

    serializable = {
        f"{lamb:.0e}": curve
        for lamb, curve in psnr_curves.items()
    }
    out_dir = f"results/classic/tv_sigma_{noise_level}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/psnr_curves.json", "w") as fp:
        json.dump(serializable, fp, indent=2)


    return best_ssim, best_lamb, best_psnr

if __name__ == "__main__":
    
    # steps = [1e-4, 1e-5, 1e-6]
    lambs = [10e0, 5e0, 2e0, 1e0, 1e-1, 1e-2, 1e-3]

    image_path_no_noise = "results/walnut_no_noise.pt"
    best_ssim, best_lamb_no, best_psnr_no = classic_TV_solver(lambs, image_path_no_noise, noise_level="none")

    print(f"\nBest PSNR without noise: {best_psnr_no:.2f} dB with ssim={best_ssim} and lambda={best_lamb_no}")

    image_path_low_noise = "results/walnut_low_noise.pt"
    best_ssim, best_lamb_low, best_psnr_low = classic_TV_solver(lambs, image_path_low_noise, noise_level="low")

    print(f"\nBest PSNR with low noise: {best_psnr_low:.2f} dB with ssim={best_ssim} and lambda={best_lamb_low}")

    image_path_high_noise = "results/walnut_high_noise.pt"
    best_ssim, best_lamb_high, best_psnr_high = classic_TV_solver(lambs, image_path_high_noise, noise_level="high")

    print(f"\nBest PSNR with high noise: {best_psnr_high:.2f} dB with ssim={best_ssim} and lambda={best_lamb_high}")

