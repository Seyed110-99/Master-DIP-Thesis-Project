import zipfile
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

zip_path = 'data/2DeteCT_slices1-1000_RecSeg.zip'
z = zipfile.ZipFile(zip_path, 'r')

all_files = z.namelist()
print(f"Total files in zip: {len(all_files)}")
print("sample files:", all_files[:5])

recon_files = sorted(
    fn for fn in all_files if fn.endswith('reconstruction.tif')
)

print(f"Found {len(recon_files)} mode-2 reconstruction slices.")

slices = []

for fn in recon_files:
    with z.open(fn) as file:
        img = Image.open(file)
        img_np = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_np)
        slices.append(img_tensor)

volume = torch.stack(slices, dim=0)
print("Volume shape:", volume.shape)
z.close()

vol = volume.unsqueeze(1)
vol256 = F.interpolate(vol, size=(256, 256), mode='bilinear', align_corners=False)
volume = vol256
torch.save(volume, 'data/DeteCT_ref_slices1-1000_mode2_256.pt')
print("Saved volume to disk.")



