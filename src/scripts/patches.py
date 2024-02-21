import logging

import config
import matplotlib.pyplot as plt
import numpy as np
import paths
import torch
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from jolideco.priors import GaussianMixtureModel
from matplotlib.patches import Rectangle
from scipy import linalg

log = logging.getLogger(__name__)

random_state = np.random.RandomState(49837)
filename = get_pkg_data_filename("galactic_center/gc_msx_e.fits")

highlight_color = "red"

image_size = (128, 68)
x, y = 10, 40
cutout = (slice(y, y + image_size[1]), slice(x, x + image_size[0]))

figsize = config.FigureSizeAA(aspect_ratio=1.2)

fig = plt.figure(figsize=figsize.inch)

ratio = image_size[0] / image_size[1]
width = 1
height = width / ratio * figsize.aspect_ratio
ax = fig.add_axes([0, 1 - height, width, height])
ax.set_axis_off()

max_value = 1
data = block_reduce(fits.getdata(filename)[cutout], 2)
data = np.clip(max_value * data / data.max(), 0, np.inf)
# data = random_state.poisson(data)
norm = simple_norm(data, "asinh", asinh_a=0.01, min_cut=0, max_cut=max_value)
ax.imshow(data, origin="lower", norm=norm, cmap="viridis")

image_size = (64, 32)
patch_size = (8, 8)
stride = 6

for x in range(0, image_size[0] - stride, stride):
    for y in range(0, image_size[1] - stride, stride):
        rectangle = Rectangle(
            xy=(x + 0.5, y + 0.5),
            width=patch_size[0],
            height=patch_size[1],
            facecolor="white",
            edgecolor="none",
            alpha=0.2,
        )

        ax.add_patch(rectangle)

        outline = Rectangle(
            xy=(x + 0.5, y + 0.5),
            width=patch_size[0],
            height=patch_size[1],
            facecolor="none",
            edgecolor="white",
            alpha=0.5,
            lw=0.2,
        )
        ax.add_patch(outline)

rectangle = Rectangle(
    xy=(30.5, 12.5),
    width=patch_size[0],
    height=patch_size[1],
    facecolor="None",
    edgecolor=highlight_color,
    lw=1.5,
)
ax.add_patch(rectangle)

ax.text(x=34.5, y=21.5, s="8 x 8 Patch", color=highlight_color, ha="center")

# example_patch = data[13:21, 13:21]
example_patch = data[13:21, 31:39]

width = 0.22
ax_patch = fig.add_axes([0.05, 0.08, width, width])
ax_patch.imshow(example_patch, origin="lower", norm=norm, cmap="viridis")
ax_patch.set_title("Example Patch", size=9, y=-0.4, color=highlight_color)
ax_patch.set_xticks([])
ax_patch.set_yticks([])
for spine in ax_patch.spines.values():
    spine.set_edgecolor(highlight_color)
    spine.set_linewidth(1.5)

ax_fig = fig.add_axes([0, 0, 1, 1])
ax_fig.set_axis_off()


gmm = GaussianMixtureModel.from_registry("jwst-cas-a-v0.1")

example_patch_torch = torch.from_numpy(example_patch).float()

example_patch_torch_flat = example_patch_torch.reshape(1, 64)

patch_mean = float(example_patch.mean())

normed = gmm.meta.patch_norm(example_patch_torch_flat)
loglike = gmm.estimate_log_prob(normed)[0, :]
idx_sort = torch.argsort(loglike, descending=True)

eigenvals_max = 6

for idx, idx_gmm in enumerate(idx_sort[:3]):
    w, v = linalg.eigh(gmm.covariances_numpy[idx_gmm])

    idx_sort = np.argsort(w)[::-1]
    v = v[:, idx_sort]
    basis = v[:, :eigenvals_max]

    weights = np.matmul(basis.T, normed.numpy().T)
    cleaned = np.matmul(basis, weights)

    # covar = gmm.covariances_numpy[idx_gmm]
    # cleaned = np.matmul(normed.numpy(), covar)
    patch = cleaned * patch_mean + patch_mean

    ax_patch = fig.add_axes([0.3 + idx * (width + 0.02), 0.08, width, width])
    ax_patch.imshow(patch.reshape(patch_size), origin="lower", cmap="viridis")
    ax_patch.set_axis_off()
    ax_patch.set_title(f"$k_{{GMM}}$={idx_gmm}", size=9, y=-0.4)

plt.axis("off")
filename = paths.figures / "patches.pdf"
log.info(f"Writing {filename}")
plt.savefig(filename, facecolor="w", dpi=300)
