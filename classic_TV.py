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
from deepinv.optim.prior        import TVPrior
from deepinv.optim.optimizers   import optim_builder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def classic_TV_solver(steps, lambs, image_path, noise_level = "none"):
    
    angles_torch = torch.linspace(0, 180, 60, device=device)

    if noise_level == "none":
        sigma = 0.0

    if noise_level == "low":
        sigma = 0.9

    else:
        sigma = 5.0


    physics = dinv.physics.Tomography(
        img_width=256, 
        angles=angles_torch, 
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
        )
    walnut_data_noisy = torch.load(image_path, map_location=device)
    walnut_data = walnut_data_noisy.to(device)

    walnut_GT = torch.load("walnut.pt", map_location=device)

    data_fidelity = L2()
    prior = TVPrior(n_it_max=20)

    # Algorithm parameters
    verbose = True
    max_iter = 10000
    early_stop = True

    best_psnr = 0.0
    best_lamb = None
    best_stepsize = None

    psnr_curves = {}

    for stepsize in steps:
        for lamb in lambs:
            print(f"Running TV-PGD with step size {stepsize} and lambda {lamb} for noise level {noise_level}...")

            params_algo = {"stepsize": stepsize, "lambda": lamb}

           
            
            model = optim_builder(
                iteration = "PGD",
                prior = prior,
                data_fidelity = data_fidelity,
                early_stop=early_stop,
                max_iter=max_iter,
                verbose=verbose,
                params_algo=params_algo,
            )

            x_model, metrics = model(
                walnut_data,
                physics = physics,
                x_gt = walnut_GT,
                compute_metrics=True,
            )

            rec_np = x_model.squeeze().detach().cpu().numpy()
            plt.imshow(rec_np, cmap='gray', vmin=0, vmax=1)
            plt.title(f"TV‐PGD σ={sigma}, λ={lamb:.1e}, stepsize={stepsize:.1e}, PSNR={metrics['psnr'][-1][-1]:.2f} dB for noise level {noise_level}")
            plt.axis('off')
            os.makedirs(f"results/classic/tv_sigma_{noise_level}", exist_ok=True)
            plt.savefig(f"results/classic/tv_sigma_{noise_level}/rec_lambda_{lamb:.0e}_step_size{stepsize:.0e}.png", dpi=200)
            plt.close()

            print(metrics['psnr'][-1][0])
            psnr_curve = metrics['psnr'][-1] # Get the PSNR curve from the metrics
            # store the full PSNR-vs-iteration curve for this lambda
            psnr_curves[lamb] = psnr_curve

            # Calculate PSNR
            PSNR = dinv.metric.PSNR(max_pixel=torch.max(walnut_GT).item())(walnut_GT, x_model).item()
            print(f"PSNR for step size {stepsize} and lambda {lamb}: {PSNR:.2f} dB for noise level {noise_level}")

            final_psnr = psnr_curve[-1]
            print(f"Final PSNR for λ={lamb:.1e}: {final_psnr:.2f} dB for noise level {noise_level}")

            if final_psnr > best_psnr:
                best_psnr = final_psnr
                best_lamb = lamb
                best_stepsize = stepsize
                print(f"New best: PSNR={best_psnr:.2f} dB lambda ={best_lamb:.2f} for noise level {noise_level} with stepsize={best_stepsize}")
        
    # 2) Plot PSNR‐vs‐iterations for each λ
    plt.figure(figsize=(6,4))
    for lamb, curve in psnr_curves.items():
        plt.plot(curve, label=f"λ={lamb:.0e}")
    plt.xlabel("PGD iteration")
    plt.ylabel("PSNR [dB]")
    plt.title(f"PSNR trajectories (noise={sigma}), best lambda={best_lamb:.0e}, best step size={best_stepsize:.0e} and highest PSNR={best_psnr:.2f} dB for noise level {noise_level}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/classic/tv_sigma_{noise_level}/psnr_trajectories.png", dpi=200)
    plt.close()

    return best_stepsize, best_lamb, best_psnr

if __name__ == "__main__":
    
    steps = [1e-4, 1e-5, 1e-6]
    lambs = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    image_path_no_noise = "results/walnut_no_noise.pt"
    best_stepsize_no, best_lamb_no, best_psnr_no = classic_TV_solver(steps, lambs, image_path_no_noise, noise_level="none")

    print(f"\nBest PSNR without noise: {best_psnr_no:.2f} dB with stepsize={best_stepsize_no} and lambda={best_lamb_no}")

    image_path_low_noise = "results/walnut_low_noise.pt"
    best_stepsize_low, best_lamb_low, best_psnr_low = classic_TV_solver(steps, lambs, image_path_low_noise, noise_level="low")

    print(f"\nBest PSNR with low noise: {best_psnr_low:.2f} dB with stepsize={best_stepsize_low} and lambda={best_lamb_low}")

    image_path_high_noise = "results/walnut_high_noise.pt"
    best_stepsize_high, best_lamb_high, best_psnr_high = classic_TV_solver(steps, lambs, image_path_high_noise, noise_level="high")

    print(f"\nBest PSNR with high noise: {best_psnr_high:.2f} dB with stepsize={best_stepsize_high} and lambda={best_lamb_high}")
    
