# %% Example Usage
import os

import numpy as np
from matplotlib import pyplot as plt

from fdx import finite_differences_grid as Ω

# Create a 2D grid
grid = Ω.Grid2d(
    xa=-1.0,
    xb=1.0,
    nx=128,
    bcx=Ω.BoundaryCondition.DIRICHLET,
    ya=-1.0,
    yb=1.0,
    ny=128,
    bcy=Ω.BoundaryCondition.DIRICHLET,
    scheme=Ω.FiniteDifferenceScheme.COMPACT,
    verbose=False,
)
X, Y = np.meshgrid(grid.x, grid.y)
u = 1.0 * np.exp(-((X - 0.5) ** 2) - (Y - 0.5) ** 2)
Du = grid.Grad(u)
Laplacian = grid.Laplacian(u)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(u, cmap="jet")
axes[0].set_title("Function")
axes[1].imshow(Du[0], cmap="jet")
axes[1].set_title("Gradient in x-direction")
axes[2].imshow(Laplacian, cmap="jet")
axes[2].set_title("Laplacian")
fig.colorbar(axes[0].imshow(u, cmap="jet"), ax=axes[0])
fig.colorbar(axes[1].imshow(Du[0], cmap="jet"), ax=axes[1])
fig.colorbar(axes[2].imshow(Laplacian, cmap="jet"), ax=axes[2])
plt.show()

# Save the figure
PATH = "../figures/introduction"
os.makedirs(PATH, exist_ok=True)
fig.savefig(os.path.join(PATH, "Gaussian_field.png"), bbox_inches="tight")
