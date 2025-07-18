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
import skimage 
from Model_arch_reg import Net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
def ellipses_DIP_dl(lambs, noise_level = "none", model_type = "ellipses", input_type = "z", critic_noise = "low"):

    
    walnut_GT = torch.load("results/walnut.pt", map_location=device)
    walnut_GT = walnut_GT.to(device)
    max_walnut = torch.max(walnut_GT).item()

    if noise_level == "none":
        walnut_data = torch.load("results/walnut_no_noise.pt", map_location=device)
    elif noise_level == "low":
        walnut_data = torch.load("results/walnut_low_noise.pt", map_location=device)
    elif noise_level == "high":
        walnut_data = torch.load("results/walnut_high_noise.pt", map_location=device)
    elif noise_level == "very_high":
        walnut_data = torch.load("results/walnut_very_high_noise.pt", map_location=device)
    else:
        raise ValueError(f"Unknown noise level {noise_level}")

    walnut_data = walnut_data.to(device)
    Height, Width = walnut_GT.shape[-2], walnut_GT.shape[-1]

    best_psnr = -float("inf")
    best_lamb = lambs[0]
    best_ssim = -float("inf")

    worst_psnr = float("inf")
    worst_ssim = float("inf")

    psnr_curves = {lamb: [] for lamb in lambs}
    ssim_curves = {lamb: [] for lamb in lambs}
    physics_new = OperatorModule(physics_raw)

    if input_type == "z":
        x_in = torch.randn((1,1,Height,Width), device=device)
        critic = Net(256, 1).to(device)
        if critic_noise == "low":
            critic.load_state_dict(torch.load("checkpoints/pre_model_reg_FBP_low.pth", map_location=device))
        elif critic_noise == "high":
            critic.load_state_dict(torch.load("checkpoints/pre_model_reg_FBP_high.pth", map_location=device))

    elif input_type == "FBP":
        x_in = physics_raw.A_dagger(walnut_data)
        critic = Net(256, 1).to(device)
        if critic_noise == "low":
            critic.load_state_dict(torch.load("checkpoints/pre_model_reg_FBP_low.pth", map_location=device))
        elif critic_noise == "high":
            critic.load_state_dict(torch.load("checkpoints/pre_model_reg_FBP_high.pth", map_location=device))
        else:
            raise ValueError(f"Unknown critic noise level {critic_noise}")
    
        
    elif input_type == "BP":
        
        x_in = physics_raw.A_adjoint(walnut_data)
        mean_x = x_in.mean()
        std_x = x_in.std()
        x_in = (x_in - mean_x) / (std_x + 1e-10) 
        x_in = torch.clamp(x_in, 0, 1)
        critic = Net(256, 1).to(device)
        critic.load_state_dict(torch.load("checkpoints/pre_model_reg_BP.pth", map_location=device))

        if critic_noise == "low":
            critic.load_state_dict(torch.load("checkpoints/pre_model_reg_BP_low.pth", map_location=device))
        elif critic_noise == "high":
            critic.load_state_dict(torch.load("checkpoints/pre_model_reg_BP_high.pth", map_location=device))
        else:
            raise ValueError(f"Unknown critic noise level {critic_noise}")
        
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
            
            if input_type == "BP":
                x_critic = x_pred - x_pred.mean()/ (x_pred.std() + 1e-10)  
                x_critic = torch.clamp(x_critic, 0, 1)
            else:
                x_critic = x_pred

            with torch.no_grad():
                adv = critic(x_critic)
                # adv = adv.mean()

            loss = mse + lamb * adv
            loss.backward()
            optimizer.step()

            psnr_value = dinv.metric.PSNR(max_pixel=max_walnut)(x_pred, walnut_GT).item()
            # append this epoch’s PSNR
            psnr_curves[lamb].append(psnr_value)

            x_pred_np = x_pred.squeeze().detach().cpu().numpy()
            x_GT_np = walnut_GT.squeeze().detach().cpu().numpy()
            
            ssim_value = skimage.metrics.structural_similarity(
                x_pred_np, 
                x_GT_np, 
                data_range= x_GT_np.max() - x_GT_np.min()
                )
            
            ssim_curves[lamb].append(ssim_value)

            if epoch % 100 == 0:
                print(f"Model type: {model_type}, Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}, Model Input: {input_type}, Noise:{noise_level}")

            if psnr_value > best_psnr and ssim_value > best_ssim:
                best_psnr = psnr_value
                best_ssim = ssim_value
                best_lamb = lamb
                best_x_pred = x_pred

            if psnr_value < worst_psnr and ssim_value < worst_ssim:
                worst_psnr = psnr_value
                worst_ssim = ssim_value
                worst_x_pred = x_pred
                worst_lamb = lamb

        x_pred_np = x_pred.squeeze().detach().cpu().numpy()
        plt.imshow(x_pred_np, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Model type: {model_type}, Model Input: {input_type}, λ={lamb:.1e}, PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}", fontsize=10)
        plt.axis('off')
        os.makedirs(f"results/DIP_dl_critic/{model_type}/{noise_level}", exist_ok=True)
        plt.savefig(f"results/DIP_dl_critic/{model_type}/{noise_level}/rec_epoch_{input_type}_{lamb:.1e}.png", dpi=200)
        plt.close()

    print(f"Best PSNR: {best_psnr:.2f}, Best SSIM: {best_ssim:.4f}, dB for λ={best_lamb:.1e}")

    best_x_pred_np = best_x_pred.squeeze().detach().cpu().numpy()
    plt.imshow(best_x_pred_np, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Model type: {model_type}, Model Input: {input_type}, λ={best_lamb:.1e}, PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f}")
    plt.axis('off')
    plt.savefig(f"results/DIP_dl_critic/{model_type}/{noise_level}/rec_epoch_{input_type}_{best_lamb:.1e}_best.png", dpi=200)
    plt.close()

    worst_x_pred_np = worst_x_pred.squeeze().detach().cpu().numpy()
    plt.imshow(worst_x_pred_np, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Model type: {model_type}, Model Input: {input_type}, λ={worst_lamb:.1e}, PSNR: {worst_psnr:.2f} dB, SSIM: {worst_ssim:.4f}")
    plt.axis('off')
    plt.savefig(f"results/DIP_dl_critic/{model_type}/{noise_level}/rec_epoch_{input_type}_{worst_lamb:.1e}_worst.png", dpi=200)
    plt.close()

    out_dir = f"results/DIP_dl_critic/{model_type}/{noise_level}/{input_type}"
    os.makedirs(out_dir, exist_ok=True)

   
    json_psnr = {f"{l:.0e}": psnr_curves[l] for l in lambs}
    json_ssim = {f"{l:.0e}": ssim_curves[l] for l in lambs}
    with open(f"{out_dir}/psnr_curves.json","w") as f:
        json.dump(json_psnr, f, indent=2)
    with open(f"{out_dir}/ssim_curves.json","w") as f:
        json.dump(json_ssim, f, indent=2)



    best_psnr_curve = psnr_curves[best_lamb]
    best_ssim_curve = ssim_curves[best_lamb]

    return best_lamb, best_psnr, best_psnr_curve, best_ssim_curve, best_ssim

if __name__ == "__main__":
    # critic_noise = ["high", "low"]
    critic_noises = ["high"]
    models = ["unet", "ellipses", "disk"]
    noise_levels = ["very_high", "none", "low", "high"]
    input_types = ["z", "FBP", "BP"]
    lambs = [50, 10, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4]
    
    sigma_max = 1.1
    white = torch.randn(1, 1, 256, 256, device=device) * sigma_max
    At_white = physics_raw.A(white)
    At_white = physics_raw.A_adjoint(At_white)
    lambda_adv = 2 * (sigma_max ** 2) * At_white.sum().item()
    lambs.append(lambda_adv)

    for critic_noise in critic_noises:
        for model_type in models:
            for noise_level in noise_levels:
                # collect best‐lambda curves for each input type
                best_curves = {}
                best_psnrs = {}
                best_lambdas = {}
                best_ssims = {}
                best_ssims_curves = {}

                for input_type in input_types:
                    best_lamb, best_psnr, best_curve, best_ssim_curve, best_ssim = ellipses_DIP_dl(
                        lambs,
                        noise_level=noise_level,
                        model_type=model_type,
                        input_type=input_type,
                        critic_noise=critic_noise
                    )
                    best_curves[input_type] = best_curve
                    best_psnrs[input_type] = best_psnr
                    best_lambdas[input_type] = best_lamb
                    best_ssims_curves[input_type] = best_ssim_curve
                    best_ssims[input_type] = best_ssim

                # find which input_type had the overall highest PSNR
                winner = max(best_psnrs.items(), key=lambda kv: kv[1])
                best_input, top_psnr = winner
                top_lambda = best_lambdas[best_input]
                top_ssim = best_ssims[best_input]

                print(f"Best input type for {model_type} with {noise_level} noise: {best_input} with PSNR={top_psnr:.2f} dB, SSIM={top_ssim:.4f} and λ={top_lambda:.1e}")

                # now plot all three on one figure
                plt.figure(figsize=(6,4))
                for input_type, curve in best_curves.items():
                    plt.plot(curve, label=input_type)
                plt.xlabel("Iterations")
                plt.ylabel("PSNR [dB]")
                plt.title(
                    f"{model_type} ({noise_level} noise), λ={top_lambda:.1e}, PSNR={top_psnr:.2f} dB"
                )
                plt.legend()
                out_dir = f"results/DIP_dl_critic/{model_type}/{noise_level}"
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(f"{out_dir}/psnr_compare_{model_type}_{noise_level}.png", dpi=200)
                plt.close()

                # Save the best SSIM curve
                plt.figure(figsize=(6,4))
                for input_type, curve in best_ssims_curves.items():
                    plt.plot(curve, label=input_type)
                plt.xlabel("Iterations")
                plt.ylabel("SSIM")
                plt.title(
                    f"{model_type} ({noise_level} noise), λ={top_lambda:.1e}, SSIM={top_ssim:.4f}"
                )
                plt.legend()
                plt.savefig(f"{out_dir}/ssim_compare_{model_type}_{noise_level}.png", dpi=200)
                plt.close()

    print("All models processed successfully.")