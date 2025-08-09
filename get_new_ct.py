import zipfile
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import urllib.request
from skimage.transform import resize

zip_path = 'data/2DeteCT_slices2001-3000_RecSeg.zip'

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the file if it doesn't exist
if not os.path.exists(zip_path):
    print(f"Downloading {zip_path}...")
    try:
        url = "https://zenodo.org/records/8017612/files/2DeteCT_slices2001-3000_RecSeg.zip?download=1" 
        
        # Download with progress indication
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
        print("\nDownload completed!")
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        print("Please download the file manually and place it at:", zip_path)
        exit(1)

z = zipfile.ZipFile(zip_path, 'r')

all_files = z.namelist()
print(f"Total files in zip: {len(all_files)}")
print("sample files:", all_files[:5])

recon_files = sorted(
    fn for fn in all_files if fn.endswith('reconstruction.tif')
)

print(f"Found {len(recon_files)} mode-2 reconstruction slices.")

# Process just the first slice
ct_img = recon_files[0]


with z.open(ct_img) as file:
    img = Image.open(file)
    img_np = np.array(img, dtype=np.float32)
    img_np = resize(img_np, (256, 256), anti_aliasing=True)
    img_tensor = torch.from_numpy(img_np)

z.close()

plt.imshow(img_tensor, cmap='gray')
plt.axis('off')
plt.savefig('results/ct_slice.png', bbox_inches='tight')

# Just add dimensions instead of stacking
volume = img_tensor.unsqueeze(0)  # Add batch dimension [1, H, W]

volume = volume.unsqueeze(1)  # Add channel dimension [1, 1, H, W]
torch.save(volume, 'results/ct.pt')
print("Saved volume to disk.")



