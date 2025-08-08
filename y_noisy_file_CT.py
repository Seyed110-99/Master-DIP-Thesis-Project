import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys
import deepinv as dinv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def image_noise_save(image_path, sigma = 0.0, name = "no_noise"):
    angles_torch = torch.linspace(0,180,60,device=device)

    ct_data = torch.load(image_path, map_location=device)
    ct_data = ct_data.to(device)
    print("Loaded CT data with shape:", ct_data.shape)

    os.makedirs("results", exist_ok=True)

    physics_raw = dinv.physics.Tomography(
        img_width=256, 
        angles=angles_torch, 
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )

    sinogram = physics_raw(ct_data)

    ct_fbp = physics_raw.A_dagger(sinogram)

    ct_bp = physics_raw.A_adjoint(sinogram)

    ct_fbp_np = ct_fbp.squeeze().detach().cpu().numpy()
    sinogram_np = sinogram.squeeze().detach().cpu().numpy()
    ct_bp_np = ct_bp.squeeze().detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    PNSR = dinv.metric.PSNR(max_pixel=torch.max(ct_data).item())(ct_data, ct_fbp).item()
    print(f"PSNR for {name}: {PNSR:.2f} dB")

    # Display the image of FBP reconstruction and sinogram

    # Display the FBP
    ax1.imshow(ct_fbp_np, cmap='gray')
    ax1.set_title(f"FBP({name}), PSNR: {PNSR:.2f} dB")
    ax1.axis('off')

    # Display the sinogram
    ax2.imshow(sinogram_np.T, cmap='gray', aspect='auto')
    ax2.set_title(f"Sinogram ({name})",)
    ax2.set_xlabel("Projection index")
    ax2.set_ylabel("Detector pixel")
    ax2.axis('off')

    ax3.imshow(ct_bp_np, cmap='gray')
    ax3.set_title(f"BP({name})")
    ax3.axis('off')
    

    plt.suptitle(f"CT FBP and Sinogram Comparison ({name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/ct_{name}_comparison.png")

    torch.save(sinogram, f"results/ct_{name}.pt")

    return sinogram

if __name__ == "__main__":
    # Define the noise levels and corresponding names
    names = ["no_noise", "low_noise", "high_noise", "very_high_noise"]
    sigmas = [0.0, 0.005, 0.009, 0.01]
    sinograms = []
    for name, sigma in zip(names, sigmas):
        image_path = "results/ct.pt"
        print(f"Processing {name} with sigma {sigma}...")
        # image_noise_save(image_path, sigma=sigma, name=name)
        sinogram = image_noise_save(image_path, sigma=sigma, name=name)
        sinograms.append(sinogram)
    print("Sinograms saved for all noise levels.")

    # sinograms = [s.squeeze().detach().cpu().numpy() for s in sinograms]
    # subtract_sinogram = sinograms[-1] - sinograms[0]
    # plt.close()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(subtract_sinogram, cmap='gray')
    # plt.title("Difference between high noise and no noise sinograms")
    # plt.axis('off')
    # plt.savefig("results/sinogram_difference.png")
    