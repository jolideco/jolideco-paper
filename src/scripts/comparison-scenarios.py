import config
import numpy as np
import paths
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

scenarios = {
    "point1": {"max_cut": 340.0, "asinh_a": 0.05},
    "aster3": {"max_cut": 200.0, "asinh_a": 0.05},
    "disk3": {"max_cut": 200.0, "asinh_a": 0.05},
    "spiral3": {"max_cut": 100.0, "asinh_a": 0.05},
}

bg = "bg1"
instrument = "chandra"

methods = [
    "gt",
    "pylira",
    "jolideco-uniform-prior=n=10",
    "jolideco-uniform-prior=n=1000",
    "jolideco-patch-prior-gleam-v0.1",
    "jolideco-patch-prior-zoran-weiss",
]

titles = {
    "gt": "Ground Truth",
    "pylira": "Pylira",
    "jolideco-uniform-prior=n=10": "Jolideco\n(uniform, n=10)",
    "jolideco-uniform-prior=n=1000": "Jolideco\n(uniform, n=1000)",
    "jolideco-patch-prior-gleam-v0.1": "Jolideco\n(GLEAM v0.1)",
    "jolideco-patch-prior-zoran-weiss": "Jolideco\n(Zoran-Weiss)",
}

figsize = config.FigureSizeAA(aspect_ratio=1.2, width_aa="two-column")


gridspec_kw = {
    "left": 0.05,
    "right": 0.95,
    "bottom": 0.05,
    "top": 0.92,
    "wspace": 0.05,
    "hspace": 0.05,
}

fig, axes = plt.subplots(
    nrows=len(scenarios),
    ncols=len(methods),
    figsize=figsize.inch,
    gridspec_kw=gridspec_kw,
)

path_base = paths.jolideco_repo_comparison / "results"

for idx, scenario in enumerate(scenarios):
    for jdx, method in enumerate(methods):
        if method == "gt":
            path = paths.jolideco_repo_comparison / "data"
            filename = list(
                path.glob(f"{instrument}_gauss_fwhm4710_128x128_mod*_{scenario}.fits")
            )[0]
        else:
            path = path_base / scenario / bg / instrument / method
            filename = path / f"{scenario}-{bg}-{method}-{instrument}-result.fits.gz"

        with fits.open(filename) as hdul:
            if method in ["pylira", "gt"]:
                data = hdul[0].data
            else:
                data = hdul["FLUX"].data

        norm_kwargs = scenarios[scenario]
        norm = simple_norm(data, stretch="asinh", min_cut=0, **norm_kwargs)
        axes[idx, jdx].imshow(data, cmap="viridis", origin="lower", norm=norm)

        if scenario == "point1":
            axes[idx, jdx].set_title(titles[method], fontsize=9)

        axes[idx, jdx].set_axis_off()


plt.savefig(paths.figures / "comparison-scenarios.pdf", dpi=config.DPI)
