import config
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import paths
from jolideco.priors import GaussianMixtureModel
from scipy import linalg

figsize = config.FigureSizeAA(aspect_ratio=1.618)

fig = plt.figure(figsize=figsize.inch)

gmm = GaussianMixtureModel.from_registry("jwst-cas-a-v0.1")


gridspec_kw = {
    "left": 0.01,
    "right": 0.99,
    "bottom": 0.005,
    "top": 0.95,
}

fig, axes = plt.subplots(
    nrows=4, ncols=8, figsize=figsize.inch, gridspec_kw=gridspec_kw
)


component = 10
covariance = gmm.covariances_numpy[component]
w, v = linalg.eigh(covariance)

idx_sort = np.argsort(w)
eigen_images = gmm.eigen_images  # v[:, idx_sort]

for idx, ax in enumerate(axes.flat):
    ax.imshow(
        eigen_images[idx].reshape((8, 8)),
        origin="lower",
        cmap="viridis",
    )
    ax.set_axis_off()
    ax.set_title(f"{idx}", size=8, pad=0.5)

plt.savefig(paths.figures / "gmm-eigen-images.pdf", facecolor="w", dpi=300)
