#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:50:06 2022

@author: bukka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:39:05 2022

@author: bukka
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:34:19 2022

@author: sandeep
"""

# Simple Allen-Cahn phase-field model
import numpy as np
from matplotlib import pyplot as plt
from sys import exit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.integrate import solve_ivp


def f_y(t, phi):
    dx = 1 / 128
    # dy = 1/512
    phi = np.reshape(phi, (257, 257))
    phi_fx = np.roll(phi, 1, axis=0)  # forward in x
    phi_bx = np.roll(phi, -1, axis=0)  # backward in x
    phi_fy = np.roll(phi, 1, axis=1)  # forward in y
    phi_by = np.roll(phi, -1, axis=1)  # backward in y
    phi_lap = (phi_fx + phi_bx + phi_fy + phi_by - 2 * phi) / (dx ** 2)

    gamma_1 = 2.5e-3  # Sort of interface energy, larger value, more diffuse, more round shapes
    mobility = 1.0  # Mobility of the interface
    gamma_2 = 1.0
    f_phi = gamma_2 * (np.power(phi, 3) - phi)

    mu = f_phi - gamma_1 * phi_lap

    mu_fx = np.roll(mu, 1, axis=0)  # forward in x
    mu_bx = np.roll(mu, -1, axis=0)  # backward in x
    mu_fy = np.roll(mu, 1, axis=1)  # forward in y
    mu_by = np.roll(mu, -1, axis=1)  # backward in y
    mu_lap = (mu_fx + mu_bx + mu_fy + mu_by - 2 * mu) / (dx ** 2)

    f_y_value = mobility * (mu_lap)

    f_y_value = np.reshape(f_y_value, -1)

    return f_y_value


# Main run part
# Initialization / geometry and parameters
N = 256  # system size in x
Ny = 2  # system size in y
Nt = 200

end_t = 1
dt = end_t / Nt

x = np.linspace(-1, 1, N + 1)
y = np.linspace(-1, 1, N + 1)

t = np.linspace(0, end_t, Nt + 1)
t_eval_v = list(np.linspace(0, end_t, Nt + 1))

X, Y = np.meshgrid(x, y, indexing="ij")

centre = 0
r = 0.4
R1 = np.sqrt((X - centre - 0.7 * r) ** 2 + (Y - centre) ** 2)
R2 = np.sqrt((X - centre + 0.7 * r) ** 2 + (Y - centre) ** 2)
epsilon = 0.1

a1 = np.tanh((r - R1) / epsilon)
a2 = np.tanh((r - R2) / epsilon)

Phi_ini = np.maximum(a1, a2)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
h1 = ax.imshow(
    Phi_ini,
    interpolation="nearest",
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
# ax.set_title('Frame = '+str(frame))
ax.set_title("$\phi$")

# Phi_ini = -np.cos(2*np.pi*x)
Phi_r = np.reshape(Phi_ini, -1)

print("starting the integration")
sol = solve_ivp(f_y, [0, 1], Phi_r, t_eval=t_eval_v)
print("end of integration")


# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# h1 = ax.imshow(sol.y, interpolation='nearest',
#         extent=[t.min(), t.max(), x.min(), x.max()],
#         origin='lower', aspect='auto')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.10)
# cbar = fig.colorbar(h1, cax=cax)
# cbar.ax.tick_params(labelsize=15)
# #ax.set_title('Frame = '+str(frame))
# ax.set_title('$\phi$')

# Phi_all = {}

# Phi_all['u'] = sol.y
# Phi_all['x'] = X
# Phi_all['t'] = T

# np.save('data/ch_1D_1e6_1e2.npy', Phi_all, allow_pickle=True)
