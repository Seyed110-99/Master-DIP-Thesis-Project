import deepinv as dinv
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import os
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
    os.makedirs("chapter1", exist_ok=True)
    x = torch.load("results/walnut.pt", map_location="cpu")
    angles = [250, 50, 25]
    noise_levels = [0.0, 0.6, 1.1]


    for i, angle in enumerate(angles):
        for j, noise in enumerate(noise_levels):
            img = make_operator(x, angle, sigma=noise)
            img = torch.clamp(img, 0, 1)
            img = (img[0,0]* 255).to(torch.uint8)
            img = Image.fromarray(img.numpy(), mode='L')
            img.save(f"chapter1/walnut_{angle}_{noise}.png")
    
    x = x.squeeze().detach().cpu().numpy()
    plt.imshow(x, cmap='gray')
    plt.axis('off')
    plt.savefig("chapter1/walnut.png", dpi=200, bbox_inches='tight')
            
    



        
        