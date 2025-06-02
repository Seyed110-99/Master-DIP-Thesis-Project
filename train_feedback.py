import torch
import os
import torch.nn as nn
from Model_arch import UNet  
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import json

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def save_image(output_tensor, iteration):
    # Convert the tensor to a numpy array
    output_image = output_tensor.squeeze().detach().cpu().numpy()
    output_image = output_image.transpose(1, 2, 0) # (C, H, W) to (H, W, C)
    output_image = np.clip(output_image, 0, 1)  # Ensure pixel values are in [0, 1]
    output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8
    plt.imsave(f"outputs/output_{iteration}.png", output_image)



def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

os.makedirs("outputs", exist_ok=True)

image = data.astronaut()
plt.imsave(f"outputs/original_image.png", image)
image = image/ 255.0

image = resize(image, (image.shape[0]/2, image.shape[1]/2), anti_aliasing=True)

noise_sigma = 0.2

noisy_image = image + noise_sigma * np.random.rand(*image.shape)
noisy_image_uint8 = (np.clip(noisy_image, 0, 1) * 255).astype(np.uint8)
plt.imsave(f"outputs/noisy_image.png", noisy_image_uint8)
noisy_image = np.clip(noisy_image, 0, 1)
noisy_tensor = torch.from_numpy(noisy_image.transpose(2, 0, 1)).float().unsqueeze(0)

psnr_noisy = calculate_psnr(image, noisy_image)

print("Noisy image PSNR: ", psnr_noisy)

input_noise = torch.rand_like(noisy_tensor)

noisy_tensor = noisy_tensor.to(device)
input_noise = input_noise.to(device)

model = UNet() # "Image to Image" 
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, eps=1e-6)
loss_fn = nn.MSELoss()

epochs = 6000
psnrs = []
psnrs_gt = []

best_psnr = 0
early_stop_patience = 10


model.train()
for epoch in range(epochs):
    output = model(input_noise) # || UNet(A^dagger measurements) - noisy_image ||
    loss = loss_fn(output, noisy_tensor) # prop to 1/PSNR(noisy, reco)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")
    if epoch % 200 == 0:
        output_image_np = output.squeeze().detach().cpu().numpy()
        output_image_np = output_image_np.transpose(1,2,0)

        noisy_image_np = noisy_tensor.squeeze().detach().cpu().numpy()
        noisy_image_np = noisy_image_np.transpose(1,2,0)

        psnr = calculate_psnr(noisy_image_np, output_image_np)

        psnrs.append((epoch, psnr.item()))
        save_image(output, epoch)
        
        # oracle PSNR (ground truth vs reconstruction)
        psnr_gt = calculate_psnr(image, output_image_np) 
        psnrs_gt.append((epoch, psnr_gt.item()))

        if psnr >= best_psnr:
            best_psnr = psnr
            early_stop_patience = 10
        else:
            early_stop_patience -= 1

    if early_stop_patience <= 0:
        print("Early stopping triggered.")
        break

iterations, psnr_values = zip(*psnrs)
iterations, psnrgt_values = zip(*psnrs_gt)

# TODO: Store PNSR as json/yaml/numpy (whatever works for you), not only the final image 

with open("outputs/psnr_noisy_reco.json", "w") as f:
     json.dump(psnrs, f)
with open("outputs/psnr_gt_reco.json", "w") as f:
    json.dump(psnrs_gt, f)

plt.plot(iterations, psnr_values, label="PSNR(noisy, reco)")
plt.plot(iterations, psnrgt_values, label="PSNR(gt, reco)")
plt.hlines(psnr_noisy, 0, iterations[-1], colors='r', label="PSNR(gt, noisy)")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('PSNR (dB)')
plt.title('PSNR over Iterations')
plt.grid(True)
plt.savefig('psnr_plot.png')

# TODO:
# Meeting every Thursday, 14:00 (in person)
# Create Github to share the code 
# Regularisation / Combat overfitting: 
# 1) Early Stopping (https://arxiv.org/abs/2112.06074)
# 2) Add term to loss function (share a paper) /  Add regularisation to the DIP (i.e. DIP + TV / DIP + L2 / ...)

# Look at different neural network architectures (UNet, ...)

# Experiment with different noise levels (noise_sigma=0.05, 0.1, 0.2)

# Denoising: y = Ax + noise with A = Id 
# CT: y = Ax + noise with A = Radon (https://deepinv.github.io/deepinv/api/stubs/deepinv.physics.Tomography.html)

# (share paper)
# Do not start with random weights, but with pre-trained weights on some other task (what could be a good task?)
# Change model input: input_noise vs. noisy_tensor 

# Do always X runs with a different seed to see if its reproducible
# Why use dropout? 

# Further Work:
# Pre-traine the DIP on denoising (BSD500?) and fine-tune on one observation
# Implement early stopping, see https://arxiv.org/abs/2112.06074 (https://arxiv.org/pdf/2302.10279) 