import deepinv as dinv
import matplotlib.pyplot as plt
import torch
import numpy as np

def make_operator(x, angles, sigma=0.0):
    angles = torch.linspace(0, 180, angles, device=x.device)
    physics = dinv.physics.Tomography(
        img_width=x.shape[-1],
        angles=angles,
        device=x.device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )
    
    sin = physics(x)
    img = physics.A_dagger(sin)
    return img 

if __name__ == "__main__":

    x = torch.load("results/walnut.pt", map_location="cpu")
    angles = [250, 50, 25]
    noise_levels = [0.0, 1.0, 1.5]

    fig, axs = plt.subplots(len(angles), len(noise_levels), figsize=(12, 10))

    for i, angle in enumerate(angles):
        for j, noise in enumerate(noise_levels):
            img = make_operator(x, angle, sigma=noise)
            img_np = img.squeeze().detach().cpu().numpy()
            axs[i, j].imshow(img_np, cmap='gray')
            axs[i, j].set_title(f"Angles: {angle}, Noise: {noise}")
            axs[i, j].axis('off')
            
    plt.tight_layout()
    plt.savefig("results/thesis.pdf", dpi=300)
    plt.show()
    



        
        