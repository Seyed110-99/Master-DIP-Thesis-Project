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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def nabla(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        b, c, h, w = x.shape
        u = torch.zeros((b, c, h, w, 2), device=x.device).type(x.dtype)
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] - x[:, :, :-1]
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] + x[:, :, 1:]
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] - x[..., :-1]
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] + x[..., 1:]
        return u

angles = torch.linspace(0, 180, 60, device=device)

physics_raw = dinv.physics.Tomography(
        img_width=256, 
        angles=angles, 
        device=device,
    )

class OperatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, operator, x):
        # ctx.operator saved for backward
        ctx.operator = operator
        # call the DeepInv forward operator A
        with torch.no_grad():
            out = operator.A(x)
        # no need to save x unless A is nonlinear
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # in the backward pass, apply the adjoint A_adjoint
        operator = ctx.operator
        grad_input = operator.A_adjoint(grad_output)
        # first returned None says “no gradient for operator”
        return None, grad_input

class OperatorModule(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, x):
        # call our custom Function
        return OperatorFunction.apply(self.operator, x)

    def A_dagger(self, y):
        # if you ever want to call the adjoint directly
        return self.operator.A_dagger(y)
    
def ellipses_DIP_dl(lambs, noise_level = "none", model_type = "ellipses", input_type = "z"):

    
    walnut_GT = torch.load("results/walnut.pt", map_location=device)
    walnut_GT = walnut_GT.to(device)
    max_walnut = torch.max(walnut_GT).item()

    if noise_level == "none":
        walnut_data = torch.load("results/walnut_no_noise.pt", map_location=device)
    elif noise_level == "low":
        walnut_data = torch.load("results/walnut_low_noise.pt", map_location=device)
    else:
       walnut_data = torch.load("results/walnut_high_noise.pt", map_location=device)

    walnut_data = walnut_data.to(device)
    Height, Width = walnut_GT.shape[-2], walnut_GT.shape[-1]

    best_psnr = -float("inf")
    best_lamb = lambs[0]
    psnr_curves = {lamb: [] for lamb in lambs}
    physics_new = OperatorModule(physics_raw)

    if input_type == "z":
        x_in = torch.randn((1,1,Height,Width), device=device)
    elif input_type == "FBP":
        x_in = physics_raw.A_dagger(walnut_data)
    elif input_type == "BP":
        x_in = physics_raw.A_adjoint(walnut_data)
    else:
        raise ValueError(f"Unknown input_type {input_type}")


    for lamb in lambs:

        if model_type == "ellipses":
            model = UNet(1, 1)
            ellipse_pretrained_weight = torch.load("checkpoints/pre_model.pth", map_location=device)
            model.load_state_dict(ellipse_pretrained_weight)

        elif model_type == "disk":
            model = UNet(1, 1)
            disk_pretrained_weight = torch.load("checkpoints/pre_disk_model.pth", map_location=device)
            model.load_state_dict(disk_pretrained_weight)

        else:
            model = UNet(1, 1)
            

        
        model = model.to(device)
        x_in = x_in.to(device)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        epochs = 5000

        for epoch in range(epochs):

            optimizer.zero_grad()
            
            x_pred = model(x_in)
            y_pred = physics_new.forward(x_pred)

            mse = criterion(y_pred, walnut_data)

            grad_u = nabla(y_pred)
            Dx = grad_u[..., 0]
            Dy = grad_u[..., 1]
            mag = torch.sqrt((Dx**2 + Dy**2).sum(dim=1) + 1e-10)

            loss = mse + lamb * torch.mean(mag)
            loss.backward()
            optimizer.step()

            psnr_value = dinv.metric.PSNR(max_pixel=max_walnut)(x_pred, walnut_GT).item()
            # append this epoch’s PSNR
            psnr_curves[lamb].append(psnr_value)

            if epoch % 100 == 0:
                print(f"Model type: {model_type}, Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, PSNR: {psnr_value:.2f} dB, Model Input: {input_type}, Noise:{noise_level}")
            
            if psnr_value > best_psnr:
                best_psnr = psnr_value
                best_lamb = lamb
        
        x_pred_np = x_pred.squeeze().detach().cpu().numpy()
        plt.imshow(x_pred_np, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Model type: {model_type}, Model Input: {input_type}, λ={lamb:.1e}, PSNR: {psnr_value:.2f} dB")
        plt.axis('off')
        os.makedirs(f"results/DIP_dl/{model_type}/{noise_level}", exist_ok=True)
        plt.savefig(f"results/DIP_dl/{model_type}/{noise_level}/rec_epoch_{input_type}_{lamb:.1e}.png", dpi=200)
        plt.close()

    print(f"Best PSNR: {best_psnr:.2f} dB for λ={best_lamb:.1e}")

    
    out_dir = f"results/DIP_dl/{model_type}/{noise_level}/{input_type}"
    os.makedirs(out_dir, exist_ok=True)

   
    json_curves = { f"{l:.0e}": curve for l, curve in psnr_curves.items() }
    with open(f"{out_dir}/psnr_curves.json", "w") as fp:
        json.dump(json_curves, fp, indent=2)
  


    best_psnr_curve = psnr_curves[best_lamb]

    return best_lamb, best_psnr, best_psnr_curve

if __name__ == "__main__":
    models      = ["unet", "ellipses", "disk"]
    noise_levels= ["none", "low", "high"]
    input_types = ["z", "FBP", "BP"]
    lambs       = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    for model_type in models:
        for noise_level in noise_levels:
            # collect best‐lambda curves for each input type
            best_curves = {}
            best_psnrs = {}
            best_lambdas = {}

            for input_type in input_types:
                best_lamb, best_psnr, best_curve = ellipses_DIP_dl(
                    lambs,
                    noise_level=noise_level,
                    model_type=model_type,
                    input_type=input_type
                )
                best_curves[input_type] = best_curve
                best_psnrs[input_type] = best_psnr
                best_lambdas[input_type] = best_lamb

            # find which input_type had the overall highest PSNR
            winner = max(best_psnrs.items(), key=lambda kv: kv[1])
            best_input, top_psnr = winner
            top_lambda = best_lambdas[best_input]
            print(f"Best input type for {model_type} with {noise_level} noise: {best_input} with PSNR={top_psnr:.2f} dB and λ={top_lambda:.1e}")

            # now plot all three on one figure
            plt.figure(figsize=(6,4))
            for input_type, curve in best_curves.items():
                plt.plot(curve, label=input_type)
            plt.xlabel("Epoch")
            plt.ylabel("PSNR [dB]")
            plt.title(
                f"{model_type} ({noise_level} noise) → best init={best_input}, λ={top_lambda:.1e}, PSNR={top_psnr:.2f} dB"
            )
            plt.legend()
            out_dir = f"results/DIP_dl/{model_type}/{noise_level}"
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/psnr_compare_{model_type}_{noise_level}.png", dpi=200)
            plt.close()

    print("All models processed successfully.")