import config
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import paths
from jolideco.priors import GaussianMixtureModel
from scipy import linalg

COMPONENT_IDX = [105, 77, 17]


figsize = config.FigureSizeAA(aspect_ratio=2.4)

fig = plt.figure(figsize=figsize.inch)

gmm = GaussianMixtureModel.from_registry("jwst-cas-a-v0.1")


gridspec_kw = {
    "left": 0.23,
    "right": 0.9,
    "bottom": 0.005,
    "top": 0.8,
}

fig, axes = plt.subplots(
    nrows=3, ncols=6, figsize=figsize.inch, gridspec_kw=gridspec_kw
)


component = 10


for row, gmm_idx in enumerate(COMPONENT_IDX):
    covariance = gmm.covariances_numpy[gmm_idx]
    w, v = linalg.eigh(covariance)
    idx_sort = np.argsort(w)[::-1]
    eigen_images = v[:, idx_sort].T

    axes[row, 0].text(
        x=-18,
        y=3.5,
        s=f"$k_{{GMM}}$={gmm_idx}",
        color="black",
        ha="left",
        va="center",
    )

    axes[row, -1].text(
        x=12,
        y=3.5,
        s="...",
        color="black",
        ha="right",
        va="center",
    )

    for col in range(0, 6):
        ax = axes[row, col]

        if row == 0:
            ax.set_title(
                f"{col+1}",
                color="black",
            )

        ax.imshow(
            eigen_images[col].reshape((8, 8)),
            origin="lower",
            cmap="viridis",
        )
        ax.set_axis_off()
        # ax.set_title(f"{idx}", size=8, pad=0.5)

plt.savefig(paths.figures / "gmm-eigen-images.pdf", facecolor="w", dpi=300)
