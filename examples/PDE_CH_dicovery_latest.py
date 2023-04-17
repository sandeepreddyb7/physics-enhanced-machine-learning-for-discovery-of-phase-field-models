#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:36:47 2022

@author: bukka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:45:24 2022

@author: bukka
"""

import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.data.burgers import burgers_delta
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import Siren, SineLayer, NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import PDEFIND, Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
from deepymod.analysis import load_tensorboard

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from scipy import linalg


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making dataset

log_dir = "runs/CH_Siren_20k_PDE_FIND/"
data_load = "data/ch_1D_1e6_1e2.npy"
pltname_u = log_dir + "u_plot"
pltname_input = log_dir + "u_input_plot"
pltname_l1 = log_dir + "L1_norm_plot"
pltname_l = log_dir + "loss_plot"

noise = 0.05

preprocess_kwargs = {"noise_level": noise}
num_samples = 20000
poly_order = 2
diff_order = 4
num_library = (poly_order + 1) * (diff_order + 1)


def load_data():
    array = np.load(data_load, allow_pickle=True).item()
    coords = torch.from_numpy(np.stack((array["t"], array["x"]), axis=-1)).float()
    data = torch.from_numpy(np.real(array["u"])).unsqueeze(-1).float()
    return coords, data


dataset = Dataset(
    load_data,
    preprocess_kwargs=preprocess_kwargs,
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_samples},
    device=device,
)

coords = dataset.get_coords().cpu()
data = dataset.get_data().cpu()
fig, ax = plt.subplots()
im = ax.scatter(coords[:, 0], coords[:, 1], c=data[:, 0], marker="x", s=10)
ax.set_xlabel("t")
ax.set_ylabel("x")
fig.colorbar(mappable=im)
# plt.savefig(pltname_input+'.png')
plt.show()

train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)

# network = NN(2, [50, 50, 50, 50 ], 1)
network = Siren(2, [64, 64, 64, 64], 1)


library = Library1D(poly_order=poly_order, diff_order=diff_order)


estimator = PDEFIND()
# estimator = Threshold(0.5)

sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=100, delta=1e-5)

constraint = LeastSquares()

model = DeepMoD(network, library, estimator, constraint).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), betas=(0.999, 0.999), amsgrad=True, lr=1e-5
)

train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    sparsity_scheduler,
    log_dir=log_dir,
    split=0.8,
    max_iterations=100000,
)

print(model.sparsity_masks)

print(model.estimator_coeffs())
print(model.constraint_coeffs())
print(model.constraint_coeffs(scaled=True, sparse=True))
print(model.constraint_coeffs(scaled=True, sparse=False))
print(model.constraint_coeffs(scaled=False, sparse=True))


### Inferring
array = np.load(data_load, allow_pickle=True).item()
coords_full = torch.from_numpy(np.stack((array["t"], array["x"]), axis=-1)).float()
data_full = torch.from_numpy(np.real(array["u"])).unsqueeze(-1).float()

Exact = array["u"]
u_star = Exact.flatten()[:, None]


coords_infer = coords_full.reshape([-1, coords_full.shape[-1]]).to(device)
data_infer = data_full.reshape([-1, data_full.shape[-1]])

prediction = model.func_approx(coords_infer)[0]

prediction = prediction.cpu()

prediction = prediction.detach().numpy()

prediction_re = np.reshape(prediction, (Exact.shape[0], Exact.shape[1]))


error = np.abs(prediction_re - Exact) / linalg.norm(Exact, "fro")


u_star_noisy = u_star + noise * np.std(u_star) * np.random.randn(
    u_star.shape[0], u_star.shape[1]
)

U_noisy = np.reshape(u_star_noisy, (Exact.shape[0], Exact.shape[1]))


fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131)
h1 = ax.imshow(
    prediction_re,
    interpolation="nearest",
    extent=[0, 1, -1, 1],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
# ax.set_title('Frame = '+str(frame))
ax.set_title("Prediction")

ax = fig.add_subplot(132)
h2 = ax.imshow(
    Exact, interpolation="nearest", extent=[0, 1, -1, 1], origin="lower", aspect="auto"
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Original")

ax = fig.add_subplot(133)
h3 = ax.imshow(
    error, interpolation="nearest", extent=[0, 1, -1, 1], origin="lower", aspect="auto"
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h3, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Error")

fig.tight_layout(pad=3.0)
plt.savefig(pltname_u + ".png")


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
h1 = ax.imshow(
    U_noisy,
    interpolation="nearest",
    extent=[0, 1, -1, 1],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
# ax.set_title('Frame = '+str(frame))
ax.set_title("Full noisy data")
ax.set_xlabel("t")
ax.set_ylabel("x")

ax = fig.add_subplot(122)
im = ax.scatter(coords[:, 0], coords[:, 1], c=data[:, 0], marker="x", s=10)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Sampled data:" + str(num_samples))
fig.colorbar(mappable=im)

fig.tight_layout(pad=3.0)
plt.savefig(pltname_input + ".png")


data_frame = load_tensorboard(log_dir)

coeffs = [
    data_frame.iloc[-1]["unscaled_coeffs_output_0_coeff_" + str(i)]
    for i in range(0, num_library)
]
np.save(log_dir + "learned_coeffs.npy", coeffs)


MSE_loss = data_frame["loss_mse_output_0"]
reg_loss = data_frame["loss_reg_output_0"]
total_loss = MSE_loss + reg_loss

plt.figure(figsize=(6, 6))
plt.semilogy(reg_loss, label="reg")
plt.semilogy(MSE_loss, label="MSE")
plt.semilogy(total_loss, label="total loss")
plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(pltname_l)


L1_loss = data_frame["loss_l1_output_0"]

plt.figure(figsize=(6, 6))
plt.semilogy(L1_loss)
plt.xlabel("Epoch")
plt.ylabel("L1 Norm")
plt.savefig(pltname_l1)
