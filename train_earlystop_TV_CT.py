import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
from Model_arch import UNet  
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
import deepinv as dinv
from skimage.transform import iradon

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


def nabla(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        b, c, h, w = x.shape
        u = torch.zeros((b, c, h, w, 2), device=x.device).type(x.dtype)
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] - x[:, :, :-1]
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] + x[:, :, 1:]
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] - x[..., :-1]
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] + x[..., 1:]
        return u

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
    plt.imsave(f"outputs/CT/output_{iteration}.png", out)

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

    os.makedirs("outputs/CT", exist_ok=True)
    # Load + preprocess the “astronaut” image
    image = data.astronaut()
    plt.imsave("outputs/CT/original_image.png", image)
    image = image / 255.0  # float in [0,1]

    # Resize to half dimensions (integer)
    h, w, _ = image.shape
    image = resize(image, (h // 2, w // 2), anti_aliasing=True)
    H, W, _ = image.shape

    
    
    # Plotting Dx and Dy
    x_true = torch.from_numpy(image[:,:,0]).float()[None, None].to(device)
    angles_torch = torch.linspace(0, 180, 60, dtype = torch.float32, device=device)  # 60 angles from 0 to 180 degrees
    
    physics = dinv.physics.Tomography(
        img_width = W,
        angles = angles_torch,
        device = device,
        noise_model = dinv.physics.GaussianNoise(sigma = 0.02)
    )
    with torch.no_grad():
        y = physics(x_true)
    y_np = y.squeeze().cpu().numpy().T  # [H,W]
    plt.figure(figsize = (6, 4))
    plt.imshow(y_np, aspect='auto', cmap='gray')
    plt.xlabel("Projection angle (degrees)")
    plt.ylabel("Detector index")
    plt.title("Sinogtram y = A(x_true) + noise")
    plt.colorbar(label="Measured intensity")
    plt.tight_layout()
    plt.savefig("outputs/CT/sinogram.png")
    plt.close()

    grad_u = nabla(x_true)  # [1,1,H,W,2]
    Dx = grad_u[0,0,...,0].cpu().numpy()  # [H,W]
    Dy = grad_u[0,0,...,1].cpu().numpy()  # [H,W]

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))

    ax1.imshow(Dx, cmap='gray'); ax1.set_title('Dₓ (vertical edges)'); ax1.axis('off')
    ax2.imshow(Dy, cmap='gray'); ax2.set_title('Dᵧ (horizontal edges)'); ax2.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/CT/gradients.png")
    plt.close(fig)

    gt_np  = x_true.squeeze().cpu().numpy()          # NumPy [H,W]

    plt.figure(figsize = (5, 5))
    plt.imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title("X_true (ground truth)")
    plt.tight_layout()
    plt.savefig("outputs/CT/x_ground_truth.png")
    plt.close()

    y_np   = y.squeeze().cpu().numpy()      
    angles_np = np.linspace(0, 180, 60, endpoint=False).astype(np.float32)
    fbp_np = iradon(y_np, theta=angles_np, filter_name='ramp', circle=True, output_size=H)
    psnr_fbp = calculate_psnr(gt_np, fbp_np)
    print(f"FBP baseline PSNR: {psnr_fbp:.2f} dB")



    # noisy_tensor = noisy_tensor.to(device)
    # input_noise  = input_noise.to(device)
    z = torch.randn(1, 1, H, W, device=device) * 0.1  # random noise tensor [1,3,H,W]

    # Build UNet and move to device
    model = UNet(in_ch=1, out_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, eps = 1e-6)
    loss_fn   = nn.MSELoss()

    epochs = 15000
    psnrs    = []                   # will hold (epoch, PSNR(noisy→recon))
    psnrs_gt = []                   # will hold (epoch, PSNR(gt→recon))
    var_history = [None] * epochs   # store EMV at each epoch

    # Instantiate ES‐WMV early‐stopper (sliding‐window size=W, patience=P)
    W = 100
    P = 10000
    
    es_wmv = EarlyStopWMV(size=W, patience=P)
    best_output = None

    model.train()
    for epoch in range(epochs):
        # ——————————————————————————————
        # 3) DIP forward + backward update
        # ——————————————————————————————
        recon = model(z)       # [1,1,H,W]


        y_pred = physics.A(recon)
        mse_loss = loss_fn(y_pred, y)  # MSE loss between noisy and predicted

        grad_u = nabla(recon)
        Dx = grad_u[..., 0]
        Dy = grad_u[..., 1]
        mag = torch.sqrt((Dx**2 + Dy**2).sum(dim=1) + 1e-10) # avoid division by zero
        TV_val = mag.mean()
        lambda_tv = 2e-5 # TV regularization weight

        # Calculate total loss: MSE + TV regularization
        
        loss   = mse_loss + lambda_tv * TV_val  # MSE + TV regularization
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f} , TV: {TV_val:.4f}")

        # ——————————————————————————————
        # 4) Compute & store PSNR every epoch
        # ——————————————————————————————
        rec_np = recon.squeeze().detach().cpu().numpy()

        psnr_gt = calculate_psnr(gt_np, rec_np)
        psnrs_gt.append((epoch, psnr_gt))

        # ——————————————————————————————
        # 5) Save a PNG every 500 epochs (unchanged)
        # ——————————————————————————————
        if epoch % 1000 == 0:
            save_image(torch.cat([recon]*3, dim=1), epoch)

        # ——————————————————————————————
        # 6) ES‐EMV logic: 
        #    (a) Let update_av handle epoch==0 initialisation
        #    (b) Then do check_stop, maybe record best_output, maybe break
        # ——————————————————————————————

        # 4) ES‐WMV logic: sliding‐window variance on the single‐channel recon

        # recon is [1,1,H,W], so squeeze to [H,W].
        rec_cpu = recon.squeeze().detach().cpu().numpy()  # [H,W]
        es_wmv.update_buffer(rec_cpu)

        # 4b) Once buffer is full, compute variance and check stop:
        if len(es_wmv.buffer) == W:
            cur_var = es_wmv.compute_variance()

            
            # store for plotting later (optional)
            var_history[epoch] = cur_var  

            # Check if we should stop:
            should_stop = es_wmv.check_stop(cur_var, epoch)

            # If this step gave a brand‐new best variance, save the reconstruction:
            if es_wmv.best_epoch == epoch:
                best_output = recon.detach().cpu().clone()
                #print(f"ES‐WMV updated best_output at epoch {es_wmv.best_epoch}, windowed variance = {es_wmv.best_score:.4f}")
            # If patience has run out, break out now:
            if should_stop:
                print(f"ES‐WMV early stop at epoch {epoch}, best_epoch = {es_wmv.best_epoch}")
                break

    # ——————————————————————————————
    # 7) After training: use best_output (or fallback to last epoch)
    # ——————————————————————————————
    if best_output is not None:
        # recon_best = best_output[:, :1, ...] #[1, 1, H, W]
        save_image(best_output.repeat(1,3,1,1), f"best_epoch_{es_wmv.best_epoch}")
        gt_np = x_true.squeeze().cpu().numpy()
        rec_np = best_output.squeeze().cpu().numpy()
        print("Final CT‐DIP PSNR:", calculate_psnr(gt_np, rec_np))

    else:
        print("ES‐WMV never updated best_output; using last epoch’s output.")
        save_image(recon.repeat(1,3,1,1), "last_epoch")
        gt_np = x_true.squeeze().cpu().numpy()
        last_np = recon.squeeze().cpu().numpy()
        print("Final PSNR(gt→recon) at last epoch:",
              f"{calculate_psnr(gt_np, last_np):.2f} dB")


    # ——————————————————————————————
    # 8) Save PSNR curves & plot (unchanged)
    # ——————————————————————————————
    with open("outputs/CT/psnr_gt_reco.json", "w") as f:
        json.dump(list(zip(*psnrs_gt)), f)

    if psnrs_gt:
        
        iterations, gt_vals = zip(*psnrs_gt)

        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.plot(iterations, gt_vals,    label="PSNR(gt→recon)",    color='C1')
        ax1.hlines(psnr_fbp, 0, iterations[-1],
                    colors='r', label="PSNR(gt→FBP)", linestyle='--')
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

        plt.savefig("outputs/CT/psnr_plot_TV_01.png")
        plt.close()

    print("Done.")