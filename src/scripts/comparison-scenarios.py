import config
import numpy as np
import paths
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

scenarios = {
    "point1": {"max_cut": 1.0},
    "aster3": {"max_cut": 1.0},
    "disk3": {"max_cut": 1.0},
    "spiral3": {"max_cut": 1.0},
}

bg = "bg1"
instrument = "chandra"

methods = [
    "pylira",
    "jolideco-uniform-prior=n=10",
    "jolideco-uniform-prior=n=1000",
    "jolideco-patch-prior-gleam-v0.1",
    "jolideco-patch-prior-zoran-weiss",
]

figsize = config.FigureSizeAA(aspect_ratio=1.2, width_aa="two-column")

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=figsize.inch)

path_base = paths.jolideco_repo_comparison / "results"

for idx, scenario in enumerate(scenarios):
    for jdx, method in enumerate(methods):
        path = path_base / scenario / bg / instrument / method
        filename = path / f"{scenario}-{bg}-{method}-{instrument}-result.fits.gz"

        with fits.open(filename) as hdul:
            if "pylira" in method:
                data = hdul[0].data
            else:
                data = hdul["FLUX"].data

        axes[idx, jdx].imshow(data, cmap="viridis", origin="lower")
        axes[idx, jdx].set_axis_off()


plt.savefig(paths.figures / "comparison-scenarios.pdf", dpi=config.DPI)
