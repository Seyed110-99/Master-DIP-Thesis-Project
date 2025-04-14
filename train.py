import torch
import os
import torch.nn as nn
from Model_arch import UNet  
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

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

input_noise = torch.rand_like(noisy_tensor)

noisy_tensor = noisy_tensor.to(device)
input_noise = input_noise.to(device)

model = UNet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

epochs = 10000
psnrs = []

best_psnr = 0
early_stop_patience = 10

model.train()
for epoch in range(epochs):
    output = model(input_noise)
    loss = loss_fn(output, noisy_tensor)
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
        
        if psnr >= best_psnr:
            best_psnr = psnr
            early_stop_patience = 10
        else:
            early_stop_patience -= 1

    if early_stop_patience <= 0:
        print("Early stopping triggered.")
        break

iterations, psnr_values = zip(*psnrs)
plt.plot(iterations, psnr_values)
plt.xlabel('Iterations')
plt.ylabel('PSNR (dB)')
plt.title('PSNR over Iterations')
plt.grid(True)
plt.savefig('psnr_plot.png')