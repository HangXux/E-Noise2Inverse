from pathlib import Path
from dataset import np_dataset, Noise2InverseDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from models.dncnn import DnCNN
from models.unet import UNet
from transforms.rotate import Rotate
from transforms.shift import Shift


# settings
epochs = 100
batch_size = 1
output_dir = Path("weights")
train_dir = Path("data/rec_split_poisson")
num_splits = 2
strategy = "X:1"
T = Rotate(n_trans=1, random_rotate=False)

# read datasets
datasets = [np_dataset(train_dir / f"{j}/*.npy") for j in range(num_splits)]
train_ds = Noise2InverseDataset(*datasets, strategy=strategy)

print(train_ds.num_slices, train_ds.num_splits)
print(train_ds[0][0].shape)
print(train_ds[0][1].shape)

dl = DataLoader(train_ds, batch_size, shuffle=True)

net = DnCNN(bias=True).cuda()
optimizer = torch.optim.Adam(net.parameters())

output_dir.mkdir(exist_ok=True)

# training
train_epochs = max(epochs // num_splits, 1)
for i in range(train_epochs):
    print("-----epoch {}-----".format(i+1))

    for (inp, tgt) in tqdm(dl):
        inp = inp.cuda()
        tgt = tgt.cuda()

        # transform
        f_inp = net(inp)
        f_inp_rotate = T.apply(f_inp)
        tgt_rotate = T.apply(tgt)

        # joint loss
        loss = nn.functional.mse_loss(f_inp, tgt) + nn.functional.mse_loss(f_inp_rotate, tgt_rotate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss: {}".format(loss.item()))

# save the weights
torch.save(
    {"epoch": int(i), "state_dict": net.state_dict(), "optimizer": optimizer.state_dict()},
    output_dir / "weights.torch"
)





