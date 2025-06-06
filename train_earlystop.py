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
    noise_sigma = 0.1
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

    epochs = 10000
    psnrs    = []                   # will hold (epoch, PSNR(noisy→recon))
    psnrs_gt = []                   # will hold (epoch, PSNR(gt→recon))
    var_history = [None] * epochs   # store EMV at each epoch

    # Instantiate ES‐WMV early‐stopper (sliding‐window size=W, patience=P)
    W = 100
    P = 400
    
    es_wmv = EarlyStopWMV(size=W, patience=P)
    best_output = None

    model.train()
    for epoch in range(epochs):
        # ——————————————————————————————
        # 3) DIP forward + backward update
        # ——————————————————————————————
        output = model(input_noise)       # [1,3,H,W]
        loss   = loss_fn(output, noisy_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

        # ——————————————————————————————
        # 4) Compute & store PSNR every epoch
        # ——————————————————————————————
        out_np   = output.squeeze().detach().cpu().numpy().transpose(1,2,0)  # [H,W,3]
        noisy_np = noisy_tensor.squeeze().detach().cpu().numpy().transpose(1,2,0)

        # psnr    = calculate_psnr(noisy_np, out_np)
        psnr_gt = calculate_psnr(image,   out_np)
        # psnrs.append((epoch, psnr))
        psnrs_gt.append((epoch, psnr_gt))

        # ——————————————————————————————
        # 5) Save a PNG every 500 epochs (unchanged)
        # ——————————————————————————————
        # if epoch % 500 == 0:
        #     save_image(output, epoch)

        # ——————————————————————————————
        # 6) ES‐EMV logic: 
        #    (a) Let update_av handle epoch==0 initialization
        #    (b) Then do check_stop, maybe record best_output, maybe break
        # ——————————————————————————————
        # Convert “output” into NumPy array [3,H,W] in [0,1]
        #   (make sure we transpose [H,W,3] → [3,H,W])
        out_cpu = out_np.transpose(2,0,1)  

        # -------------------------------
        # 4) ES‐WMV logic: sliding‐window variance
        # -------------------------------
        # Convert “output” into NumPy [3,H,W]:
        out_cpu = out_np.transpose(2, 0, 1)

        # 4a) Push this reconstruction into our fixed-size FIFO buffer:
        es_wmv.update_buffer(out_cpu)

        # 4b) Once we have W images, compute the windowed variance:
        if len(es_wmv.buffer) == W:
            cur_var = es_wmv.compute_variance()

            # store for plotting later (optional)
            var_history[epoch] = cur_var  

            # Check if we should stop:
            should_stop = es_wmv.check_stop(cur_var, epoch)

            # If this step gave a brand‐new best variance, save the reconstruction:
            if es_wmv.best_epoch == epoch:
                best_output = output.detach().cpu().clone()

            # If patience has run out, break out now:
            if should_stop:
                print(f"ES‐WMV early stop at epoch {epoch}, best_epoch = {es_wmv.best_epoch}")
                break

    # ——————————————————————————————
    # 7) After training: use best_output (or fallback to last epoch)
    # ——————————————————————————————
    if best_output is not None:
        # Note: use es_wmv.best_epoch, not ewmvar.best_epoch
        save_image(best_output, f"eswmv_best_epoch_{es_wmv.best_epoch}")
        final_np = best_output.squeeze(0).cpu().numpy().transpose(1,2,0)
        print("Final PSNR(gt→recon) at best epoch:",
              f"{calculate_psnr(image, final_np):.2f} dB")
    else:
        print("ES‐WMV never updated best_output; using last epoch’s output.")
        last_output = output.detach().cpu().clone()
        save_image(last_output, "last_epoch")
        last_np = last_output.squeeze(0).cpu().numpy().transpose(1,2,0)
        print("Final PSNR(gt→recon) at last epoch:",
              f"{calculate_psnr(image, last_np):.2f} dB")


    # ——————————————————————————————
    # 8) Save PSNR curves & plot (unchanged)
    # ——————————————————————————————
    # with open("outputs/psnr_noisy_reco.json", "w") as f:
    #     json.dump(list(zip(*psnrs)), f)
    with open("outputs/psnr_gt_reco.json", "w") as f:
        json.dump(list(zip(*psnrs_gt)), f)

    if psnrs:
        iterations, noisy_vals = zip(*psnrs)
        _, gt_vals             = zip(*psnrs_gt)

        fig, ax1 = plt.subplots(figsize=(8,5))
        # ax1.plot(iterations, noisy_vals, label="PSNR(noisy→recon)", color='C0')
        ax1.plot(iterations, gt_vals,    label="PSNR(gt→recon)",    color='C1')
        ax1.hlines(psnr_noisy, 0, iterations[-1],
                   colors='r', label="PSNR(gt→noisy)", linestyle='--')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title("PSNR & EMV over Iterations")
        ax1.grid(True)

        ax2 = ax1.twinx()
        var_vals = [(var_history[it] if it < len(var_history) else np.nan)
                    for it in iterations]
        ax2.plot(iterations, var_vals, label="Window‐Variance", color='C2', linestyle=':')
        ax2.set_ylabel("Windowed Variance")

        ax2.tick_params(axis='y', labelcolor='C2')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.savefig("outputs/psnr_plot.png")
        plt.close()

    print("Done.")