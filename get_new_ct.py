import zipfile
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

zip_path = 'data/2DeteCT_slices2001-3000_RecSeg.zip'
z = zipfile.ZipFile(zip_path, 'r')

all_files = z.namelist()
print(f"Total files in zip: {len(all_files)}")
print("sample files:", all_files[:5])

recon_files = sorted(
    fn for fn in all_files if fn.endswith('reconstruction.tif')
)

print(f"Found {len(recon_files)} mode-2 reconstruction slices.")

ct_img = recon_files[0]


with z.open(ct_img) as file:
    img = Image.open(file)
    img_np = np.array(img, dtype=np.float32)
    img_tensor = torch.from_numpy(img_np)

z.close()

plt.imshow(img_tensor, cmap='gray')
plt.axis('off')
plt.savefig('results/ct_slice.png', bbox_inches='tight')

volume = torch.stack(img_tensor, dim=0)


vol = volume.unsqueeze(1)
vol256 = F.interpolate(vol, size=(256, 256), mode='bilinear', align_corners=False)
volume = vol256
torch.save(volume, 'results/ct.pt')
print("Saved volume to disk.")



