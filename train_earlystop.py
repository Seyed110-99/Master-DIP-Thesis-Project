import torch
import os
import torch.nn as nn
from Model_arch import UNet  
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
from collections import deque
import copy

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
    eps = 1e-10  # Small value to avoid division by zero
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(original)  # Use the maximum pixel value from the original image
    psnr = 10 * np.log10(((max_pixel**2) + eps)/ mse)
    return psnr

os.makedirs("outputs", exist_ok=True)

image = data.astronaut()
plt.imsave(f"outputs/original_image.png", image)
image = image/ 255.0

image = resize(image, (image.shape[0]/2, image.shape[1]/2), anti_aliasing=True)

noise_sigma = 0.1

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

epochs = 10000
psnrs = []
psnrs_gt = []


W = 50 # Window size 
P = 1000 # Patience for early stopping

var_min = float('inf') 
stagnant_count = 0

recent_preds = deque(maxlen=W) # Store recent predictions for early stopping
var_history = [] # Store variance history for debugging
model.train()
for epoch in range(epochs):
    output_raw = model(input_noise) # || UNet(A^dagger measurements) - noisy_image ||
    output = torch.sigmoid(output_raw)  # Ensure output is in [0, 1] range
    loss = loss_fn(output, noisy_tensor) # prop to 1/PSNR(noisy, reco)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    output_np = output.detach().cpu().squeeze().permute(1, 2, 0).numpy()  # (C, H, W) to (H, W, C)

    recent_preds.append(output_np.copy())
    print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # Calculate PSNR between noisy image and output
    output_image_np = output.squeeze().detach().cpu().numpy()
    output_image_np = output_image_np.transpose(1,2,0)

    noisy_image_np = noisy_tensor.squeeze().detach().cpu().numpy()
    noisy_image_np = noisy_image_np.transpose(1,2,0)

    psnr = calculate_psnr(noisy_image_np, output_image_np)
    psnrs.append((epoch, psnr.item()))

    # oracle PSNR (ground truth vs reconstruction)
    psnr_gt = calculate_psnr(image, output_image_np) 
    psnrs_gt.append((epoch, psnr_gt.item()))

    if len(recent_preds) == W:

        # Calculate variance of recent predictions
        stacked_preds = np.stack(recent_preds, axis=0)

        # Calculate variance across the window
        var = np.var(stacked_preds, axis=0) 

        var_mean = np.mean(var)
        print(f"Epoch [{epoch}/{epochs}], Variance: {var_mean:.4f}")
        var_history.append((epoch, var_mean))

        if var_mean < var_min:
            var_min = var_mean
            stagnant_count = 0 

            best_weights   = copy.deepcopy(model.state_dict())
            best_output_np = output_np.copy()

        else:
            stagnant_count += 1
            print(f"Epoch [{epoch}/{epochs}], Stagnation count: {stagnant_count}")
        if stagnant_count >= P:
            print(f"Early stopping triggered at epoch {epoch} due to stagnation.")
            break

    if epoch % 500 == 0:
        save_image(output, epoch)


iterations, psnr_values = zip(*psnrs)
iterations, psnrgt_values = zip(*psnrs_gt)
iterations_var, var_values = zip(*var_history)

if best_weights is not None:
    model.load_state_dict(best_weights)
    print("Loaded best model weights based on variance stagnation.")

    # Convert best_output_np to uint8 before saving
    best_uint8 = (np.clip(best_output_np, 0, 1) * 255).astype(np.uint8)
    plt.imsave("outputs/best_output.png", best_uint8)
    print("Best output saved as 'best_output.png'.")

with open("outputs/psnr_noisy_reco.json", "w") as f:
     json.dump(psnrs, f)
with open("outputs/psnr_gt_reco.json", "w") as f:
    json.dump(psnrs_gt, f)
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.plot(iterations, psnr_values, label="PSNR(noisy, reco)")
plt.plot(iterations, psnrgt_values, label="PSNR(gt, reco)")
ax1.plot(iterations,    psnr_values,   label="PSNR(noisy→reco)",  color="C0")
ax1.plot(iterations,    psnrgt_values, label="PSNR(gt→reco)",     color="C1")
ax1.axhline(psnr_noisy,
            color="r", linestyle="--",
            label="PSNR(gt→noisy)")
ax1.set_xlabel("Iteration (epoch)")
ax1.set_ylabel("PSNR (dB)")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(iterations_var, var_values, color="green", linestyle=":", label="Variance")
ax2.set_ylabel("VAR")         # you can add a scale factor if you like, e.g. “VAR ×10³”
ax2.set_ylim(0, max(var_values)*1.1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("PSNR & VAR over Iterations")
plt.tight_layout()
plt.savefig("outputs/psnr_and_var_plot.png")
plt.show()