import torch
import os
import torch.nn as nn
from Model_arch import UNet  
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import json

# ----------------------------------------
# 1) EWMVar class for early stopping
# ----------------------------------------
class EWMVar():
    """
    Exponentially-Weighted Moving Variance (EMV) early stopper.
    α: smoothing parameter (0 < α < 1),
    p: patience (number of iterations EMV can fail to decrease before stopping).
    """
    def __init__(self, alpha, p):
        self.alpha       = alpha       # smoothing factor
        self.patience    = p           # how many “no improvement” steps to wait
        self.wait_count  = 0
        self.best_emv    = float('inf')
        self.best_epoch  = 0
        self.stop        = False
        self.ema         = None        # exponential moving average
        self.emv         = None        # current EMV

    def check_stop(self, cur_epoch):
        """
        Compare current EMV to best seen so far.
        If lower, update best_emv & best_epoch, reset wait_count.
        Otherwise increment wait_count and set stop=True if patience exceeded.
        """
        if self.emv < self.best_emv:
            self.best_emv   = self.emv
            self.best_epoch = cur_epoch
            self.wait_count = 0
        else:
            self.wait_count += 1
            self.stop = (self.wait_count >= self.patience)

    def update_av(self, cur_img, cur_epoch):
        """
        Given cur_img (NumPy array in [0,1], shape=(C,H,W)) at iteration cur_epoch,
        update exponential moving average (self.ema) and compute new EMV (self.emv).
        """
        delta = cur_img - self.ema
        tmp_ema = self.ema + self.alpha * delta
        # Clip EMA back to [0,1]
        self.ema = np.clip(tmp_ema, 0.0, 1.0)
        # EMV update rule: (1−α)*(old_emv + α * ||delta||^2)
        self.emv = (1.0 - self.alpha) * (self.emv + self.alpha * (np.linalg.norm(delta) ** 2))


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


# ----------------------------------------
# 3) Main script
# ----------------------------------------
if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs("outputs", exist_ok=True)

    # Load + preprocess the “astronaut” image
    image = data.astronaut()
    plt.imsave("outputs/original_image.png", image)
    image = image / 255.0  # float in [0,1]

    # Resize to half dimensions (integer)
    h, w, _ = image.shape
    image = resize(image, (h // 2, w // 2), anti_aliasing=True)
    H, W, _ = image.shape

    # Add uniform noise in [0, noise_sigma]
    noise_sigma = 0.2
    noisy_image = image + noise_sigma * np.random.rand(H, W, 3)
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
    plt.imsave("outputs/noisy_image.png", noisy_image_uint8)

    noisy_tensor = torch.from_numpy(noisy_image.transpose(2, 0, 1)).float().unsqueeze(0)  # [1,3,H,W]
    psnr_noisy  = calculate_psnr(image, noisy_image)
    print("Noisy image PSNR: ", psnr_noisy)

    # Fixed uniform-[0,1] input noise
    input_noise = torch.rand_like(noisy_tensor)

    noisy_tensor = noisy_tensor.to(device)
    input_noise  = input_noise.to(device)

    # Build UNet and move to device
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, eps=1e-6)
    loss_fn   = nn.MSELoss()

    epochs = 6000
    psnrs    = []                   # will hold (epoch, PSNR(noisy→recon))
    psnrs_gt = []                   # will hold (epoch, PSNR(gt→recon))
    emv_history = [None] * epochs   # store EMV at each epoch

    best_psnr = 0
    early_stop_patience = 10

    # Instantiate EMV (α=0.1, patience=300)
    ewmvar = EWMVar(alpha=0.1, p=500)
    best_output = None

    # Warm‐up threshold = 200
    warmup = 200

    model.train()
    for epoch in range(epochs):
        # ------ Forward + backward, exactly like your original code ------
        output = model(input_noise)         # [1,3,H,W]
        loss   = loss_fn(output, noisy_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")
        out_np   = output.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,3]
        noisy_np = noisy_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

        psnr    = calculate_psnr(noisy_np, out_np)
        psnr_gt = calculate_psnr(image, out_np)

        psnrs.append((epoch, psnr))
        psnrs_gt.append((epoch, psnr_gt))

        # --- PSNR logging + saving every 200 epochs (unchanged) ---
        if epoch % 500 == 0:
            
            save_image(output, epoch)

            

        # ----------------------
        # 4) EMV logic, keyed off the paper’s prescription
        # ----------------------
        # a) At epoch < warmup, do nothing
        if epoch < warmup:
            emv_history[epoch] = None

        elif epoch == warmup:
            # Initialize EMA & EMV exactly at warmup
            out_cpu = output.detach().cpu().squeeze(0).numpy()
            out_cpu = np.clip(out_cpu, 0.0, 1.0)
            ewmvar.ema = out_cpu.copy()
            ewmvar.emv = 0.0
            emv_history[epoch] = ewmvar.emv

        elif epoch == warmup + 1:
            # First real variance update, but skip check_stop
            out_cpu = output.detach().cpu().squeeze(0).numpy()
            out_cpu = np.clip(out_cpu, 0.0, 1.0)
            ewmvar.update_av(cur_img=out_cpu, cur_epoch=epoch)
            emv_history[epoch] = ewmvar.emv

        else:
            # epoch >= warmup+2: update → compare → possibly save best_output
            out_cpu = output.detach().cpu().squeeze(0).numpy()
            out_cpu = np.clip(out_cpu, 0.0, 1.0)
            ewmvar.update_av(cur_img=out_cpu, cur_epoch=epoch)
            emv_history[epoch] = ewmvar.emv

            ewmvar.check_stop(cur_epoch=epoch)
            if ewmvar.best_epoch == epoch:
                best_output = output.detach().cpu().clone()
            if ewmvar.stop:
                print(f"ES-WMV early stop at epoch {epoch}, best_epoch = {ewmvar.best_epoch}")
                break

    # --------------------------------------
    # 5) After training, save the “best EMV” reconstruction (or fallback)
    # --------------------------------------
    if best_output is not None:
        save_image(best_output, f"eswmv_best_epoch_{ewmvar.best_epoch}")
        final_np = best_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        print("Final PSNR(gt→recon) at best epoch:",
              f"{calculate_psnr(image, final_np):.2f} dB")
    else:
        print("ES-WMV never updated best_output; using last epoch’s output.")
        last_output = output.detach().cpu().clone()
        save_image(last_output, "last_epoch")
        last_np = last_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        print("Final PSNR(gt→recon) at last epoch:",
              f"{calculate_psnr(image, last_np):.2f} dB")

    # --------------------------------------
    # 6) Save PSNR curves + JSON logs (unchanged)
    # --------------------------------------
    with open("outputs/psnr_noisy_reco.json", "w") as f:
        json.dump(list(zip(*psnrs)), f)
    with open("outputs/psnr_gt_reco.json", "w") as f:
        json.dump(list(zip(*psnrs_gt)), f)

    # --------------------------------------
    # 7) Plot PSNR (noisy→recon, gt→recon) + EMV on same figure
    # --------------------------------------
    if psnrs:
        iterations, noisy_vals = zip(*psnrs)
        _, gt_vals             = zip(*psnrs_gt)

        # Create PSNR figure
        fig, ax1 = plt.subplots()
        ax1.plot(iterations, noisy_vals, label="PSNR(noisy→recon)", color='C0')
        ax1.plot(iterations, gt_vals,    label="PSNR(gt→recon)",    color='C1')
        ax1.hlines(psnr_noisy, 0, iterations[-1],
                   colors='r', label="PSNR(gt→noisy)", linestyle='--')
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title("PSNR & EMV over Iterations")
        ax1.grid(True)

        # Now overlay EMV on a second y-axis
        ax2 = ax1.twinx()
        emv_vals_at_logged_iters = []
        for it in iterations:
            emv_vals_at_logged_iters.append(emv_history[it] if it < len(emv_history) else np.nan)

        ax2.plot(iterations, emv_vals_at_logged_iters, label="EMV", color='C2', linestyle=':')
        ax2.set_ylabel("EMV (exponentially-weighted variance)")
        ax2.tick_params(axis='y', labelcolor='C2')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # Save to the same filename (unchanged)
        plt.savefig("outputs/psnr_plot.png")
        plt.close()

    print("Done.")
