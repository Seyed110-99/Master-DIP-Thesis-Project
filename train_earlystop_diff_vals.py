import torch
import os
import torch.nn as nn
from Model_arch import UNet  
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
import itertools as it


# ----------------------------------------
# 1) ES‐WMV EarlyStop class (sliding‐window variance)
# ----------------------------------------
class EarlyStopWMV:
    """
    Sliding‐Window (size = W) Moving Variance early‐stopper.
      • size:     window length W
      • patience: how many consecutive iterations without variance‐improvement
                  before stopping
    """
    def __init__(self, size, patience):
        self.size       = size
        self.patience   = patience
        self.wait_count = 0
        self.best_score = float('inf')  # “smallest windowed‐variance seen so far”
        self.best_epoch = 0
        self.buffer     = []            # will hold up to `size` NumPy arrays [C,H,W]
        self.stop       = False

    def update_buffer(self, cur_img):
        """
        Append cur_img (NumPy [C,H,W]) to FIFO buffer.
        If buffer > W, pop the oldest element.
        """
        self.buffer.append(cur_img.copy())
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def compute_variance(self):
        """
        Once buffer is full (len(buffer)==W), compute:
          ave_img   = mean(buffer, axis=0)               # shape [C,H,W]
          var_list  = [ || x_i - ave_img ||_F^2  for i ]  # each scalar
          cur_var   = average of var_list
        Return cur_var as a float.
        """
        X = np.stack(self.buffer, axis=0).astype(np.float32)  # shape (W, C, H, W)
        ave_img = X.mean(axis=0)                              # shape [C,H,W]
        diffs   = X - ave_img                                 # shape (W, C, H, W)
        # Frobenius norm squared of each slice:
        var_per_frame = (diffs.reshape(self.size, -1) ** 2).sum(axis=1)  # shape (W,)
        cur_var = float(var_per_frame.mean())
        return cur_var

    def check_stop(self, cur_var, cur_epoch):
        """
        If cur_var < best_score:
          • best_score = cur_var
          • best_epoch = cur_epoch
          • wait_count  = 0
        Else:
          • wait_count += 1
          • if wait_count >= patience: self.stop = True
        Returns self.stop (True if we should break).
        """
        if cur_var < self.best_score:
            self.best_score = cur_var
            self.best_epoch = cur_epoch
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stop = True
        return self.stop
    
# ----------------------------------------
# 2) Helper functions (unchanged)
# ----------------------------------------
def save_image(output_tensor, iteration):
    """
    Save a [1,3,H,W] torch tensor (float in [0,1]) as outputs/output_{iteration}.png.
    """
    out = output_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,3]
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)
    plt.imsave(f"outputs/output_{iteration}.png", out)

def calculate_psnr(original, denoised):
    """
    Compute PSNR between two NumPy arrays in [0,1], same shape.
    """
    eps = 1e-10
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(original)
    return 10 * np.log10(((max_pixel**2) + eps) / mse)


def run_dip_with_es_wmv(noise_sigma, max_epochs, window_size, patience, device):
    """
    Run the DIP with ES-WMV early stopping.
    This function is a placeholder for the main training loop.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() 
                              else "cpu")
        
    os.makedirs("outputs", exist_ok=True)

    # 1) Load + preprocess “astronaut” (unchanged)
    image = data.astronaut()
    image = image / 255.0
    h, w, _ = image.shape
    image = resize(image, (h // 2, w // 2), anti_aliasing=True)
    H, W, _ = image.shape

    # 2) Add uniform noise in [0, noise_sigma]
    noisy_image = image + noise_sigma * np.random.rand(H, W, 3)
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_tensor = torch.from_numpy(noisy_image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    psnr_noisy  = calculate_psnr(image, noisy_image)
    print("Noisy image PSNR:", psnr_noisy)


    # Fixed uniform-[0,1] input noise
    input_noise = torch.rand_like(noisy_tensor)

    noisy_tensor = noisy_tensor.to(device)
    input_noise  = input_noise.to(device)

    # Build UNet and move to device
    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, eps=1e-6)

    loss_fn   = nn.MSELoss()

    epochs = max_epochs
    psnrs_gt = []                   # will hold (epoch, PSNR(gt→recon))
    var_history = [None] * epochs   # store EMV at each epoch

    best_output = None
    # Instantiate ES‐WMV early‐stopper (sliding‐window size=W, patience=P)
    es_wmv = EarlyStopWMV(size=window_size, patience=patience)

    model.train()
    for epoch in range(epochs):
        output = model(input_noise)       # [1,3,H,W]
        loss   = loss_fn(output, noisy_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

        #Compute PSNR every epoch
        out_np   = output.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        psnr_gt  = calculate_psnr(image,   out_np)
        psnrs_gt.append((epoch, psnr_gt))
        
        # Save an image every 500 epochs
        # if epoch % 500 == 0:
        #     save_image(output, epoch)
        
        #ES-WMV logic
        out_cpu = out_np.transpose(2, 0, 1)  # [3,H,W]
        es_wmv.update_buffer(out_cpu)

        if len(es_wmv.buffer) == window_size:

            cur_var = es_wmv.compute_variance()
            var_history[epoch] = cur_var  # store for plotting later
            
            should_stop = es_wmv.check_stop(cur_var, epoch)
            if es_wmv.best_epoch == epoch:
                best_output = output.detach().cpu().clone()
            if should_stop:
                print(f"ES‐WMV early stop at epoch {epoch}, best_epoch = {es_wmv.best_epoch}")
                break
        

    if best_output is not None:
        # Note: use es_wmv.best_epoch, not ewmvar.best_epoch
        save_image(best_output, f"eswmv_best_epoch_{es_wmv.best_epoch}_{noise_sigma}_{max_epochs}")
        final_np = best_output.squeeze(0).cpu().numpy().transpose(1,2,0)
        best_epoch = es_wmv.best_epoch
        print("Final PSNR(gt→recon) at best epoch:",
              f"{calculate_psnr(image, final_np):.2f} dB")
    else:
        print("ES‐WMV never updated best_output; using last epoch’s output.")
        last_output = output.detach().cpu().clone()
        save_image(last_output, f"last_epoch_{noise_sigma}_{max_epochs}")
        last_np = last_output.squeeze(0).cpu().numpy().transpose(1,2,0)
        print("Final PSNR(gt→recon) at last epoch:",
              f"{calculate_psnr(image, last_np):.2f} dB")
    with open(f"outputs/psnr_gt_reco_{noise_sigma}_{max_epochs}.json", "w") as f:
        json.dump(list(zip(*psnrs_gt)), f)
    iterations = [ep for ep, _ in psnrs_gt]
    gt_values = [psnr for _, psnr in psnrs_gt]

    return iterations, gt_values, var_history, psnr_noisy, best_epoch
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "mps" if torch.backends.mps.is_available() 
                                  else "cpu")
    
    noise_levels = [0.05, 0.1, 0.2]
    epoch_settings = [6000, 10000, 15000]

    fig, (ax_psnr, ax_var) = plt.subplots(1, 2, figsize =(14,5))




    for noise_sigma in noise_levels:
        for max_epochs in epoch_settings:

            rand_color = np.random.rand(3, )
            window_size = 100
            patience = 500
            
            iterations, psnr_values, var_history, psnr_noisy, best_epoch = run_dip_with_es_wmv(
                noise_sigma, max_epochs, window_size, patience, device
            )
            # Plot PSNR values
            ax_psnr.plot(iterations, psnr_values, 
                         label=f"Noise: {noise_sigma}, Epochs: {max_epochs}",
                         color = rand_color)
            ax_psnr.axhline(psnr_noisy, color=rand_color, linestyle=':', alpha=0.6)

            ax_psnr.axvline(best_epoch, color=rand_color, linestyle=':', alpha=0.6)

            # Plot Window‐Variance vs. iteration
            var_vals = [var_history[i] if (i < len(var_history) and var_history[i] is not None) else np.nan
                        for i in iterations]
            ax_var.plot(iterations, var_vals, label=f"σ={noise_sigma}, E={max_epochs}",
                color=rand_color, linestyle='--')    
            
            ax_var.axvline(best_epoch, color=rand_color, linestyle=':', alpha=0.6)

    # Finalize PSNR subplot
    ax_psnr.set_xlabel("Iteration")
    ax_psnr.set_ylabel("PSNR (gt→recon) [dB]")
    ax_psnr.set_title("PSNR (gt→recon) for Different Noise/Epoch Settings")
    ax_psnr.legend(loc='lower right', fontsize='small')
    ax_psnr.grid(True)

    # Finalize Variance subplot
    ax_var.set_xlabel("Iteration")
    ax_var.set_ylabel("Windowed Variance")
    ax_var.set_title("Sliding‐Window Variance for Different Noise/Epoch Settings")
    ax_var.legend(loc='upper right', fontsize='small')
    ax_var.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/multiple_settings_comparison.png")
    plt.close(fig)

    print("All experiments finished. See outputs/multiple_settings_comparison.png.")