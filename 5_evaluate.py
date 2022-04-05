import numpy as np
import torch
from tqdm import tqdm
from dataset import np_dataset, Noise2InverseDataset
from torch.utils.data import DataLoader
from pathlib import Path
from models.dncnn import DnCNN
import os
import skimage.metrics as skm
from models.unet import UNet
import matplotlib.pyplot as plt


num_splits = 2
strategy = "X:1"
batch_size = num_splits


input_dir = Path("data/test_split_poisson")
weights_path = Path("weights//weights.torch")
output_dir = Path("denoised")
output_dir.mkdir(exist_ok=True)

datasets = [np_dataset(input_dir / f"{j}/*.npy") for j in range(num_splits)]
ds = Noise2InverseDataset(*datasets, strategy=strategy)
dl = DataLoader(ds, batch_size, shuffle=False)

net = DnCNN(bias=False)
state = torch.load(weights_path)
net.load_state_dict(state["state_dict"])
net = net.cuda()

net.eval()
with torch.no_grad():
    for i, batch in tqdm(enumerate(dl)):
        inp, _ = batch
        inp = inp.cuda()
        out = net(inp)
        out = out.mean(dim=0)
        out_np = out.detach().cpu().numpy().squeeze()
        np.save(os.path.join(output_dir, "output"), out_np)


img_denoised = np.load("denoised/output.npy")
img_poisson = np.load("data/test_poisson.npy")
img_clean = np.load("data/full_fbp.npy")

# plot
plt.gray()
plt.subplot(131)
plt.imshow(img_clean)
plt.axis('off')
plt.subplot(132)
plt.imshow(img_poisson)
plt.axis('off')
plt.subplot(133)
plt.imshow(img_denoised)
plt.axis('off')
plt.show()

# PSNR, SSIM
clean = np.load("data/full_fbp.npy")
output = np.load("denoised/output.npy")

data_range = clean.max() - clean.min()
psnr = skm.peak_signal_noise_ratio(clean, output)
ssim = skm.structural_similarity(clean, output)

print(f"PSNR:  {psnr:5.2f}")
print(f"SSIM:  {ssim:5.2f}")



