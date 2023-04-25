import config
import numpy as np
import paths
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

asinh_a = 0.01
scenario_titles = {
    "A1": {
        "name": "point1",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
    "B3": {
        "name": "aster3",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
    "C3": {
        "name": "disk3",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
    "D4": {
        "name": "spiral4",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
}

bg = "bg1"
instrument = "chandra"


method_titles = {
    "data": "Data",
    "gt": "Ground Truth",
    "jolideco-uniform-prior=n=10": "Jolideco\n(Uni, n=10)",
    "jolideco-uniform-prior=n=1000": "Jolideco\n(Unif., n=1000)",
    "pylira": "Pylira",
    "jolideco-patch-prior-zoran-weiss": "Jolideco\n(Zoran-Weiss)",
    "jolideco-patch-prior-gleam-v0.1": "Jolideco\n(GLEAM v0.1)",
}

figsize = config.FigureSizeAA(aspect_ratio=1.618, width_aa="two-column")

upsampling_factor = 2
DATA_SHAPE = (128, 128)

gridspec_kw = {
    "left": 0.05,
    "right": 0.90,
    "bottom": 0.02,
    "top": 0.92,
    "wspace": 0.05,
    "hspace": 0.02,
}

fig, axes = plt.subplots(
    nrows=len(scenario_titles),
    ncols=len(method_titles),
    figsize=figsize.inch,
    gridspec_kw=gridspec_kw,
)


def add_cbar(im, ax, fig, label=""):
    """Add cbar to a given axis and figure"""
    bbox = ax.get_position()
    loright = bbox.corners()[-2]
    rect = [loright[0] + 0.005, loright[1], 0.02, bbox.height]
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(label, size=9)
    cbar.set_ticks([0, 10, 100])
    cax.tick_params(labelsize=9)


def read_and_stack_counts(scenario, bg, instrument):
    """Read and stack couns files"""
    path = paths.jolideco_repo_comparison / "data"

    filenames = path.glob(
        f"{instrument}_gauss_fwhm4710_128x128_sim*_{bg}_{scenario}_iter*.fits"
    )

    data_all = []

    for filename in filenames:
        data_all.append(fits.getdata(filename))

    return np.mean(data_all, axis=0)


def read_flux_ref(scenario):
    """Reda reference flux"""
    path = paths.jolideco_repo_comparison / "data"
    filenames = path.glob(f"{instrument}_gauss_fwhm4710_128x128_mod*_{scenario}.fits")
    return fits.getdata(list(filenames)[0])


def read_flux(scenario, bg, instrument):
    """Read flux image"""
    path_base = paths.jolideco_repo_comparison / "results"
    path = path_base / scenario / bg / instrument / method
    filename = path / f"{scenario}-{bg}-{method}-{instrument}-result.fits.gz"

    with fits.open(filename) as hdul:
        try:
            data = hdul["FLUX"].data
        except KeyError:
            data = hdul[0].data

    # adjust flux norm for upsampled data
    if data.shape != DATA_SHAPE:
        data *= upsampling_factor**2

    return data


for idx, scenario_title in enumerate(scenario_titles):
    scenario = scenario_titles[scenario_title]["name"]
    norm_kwargs = scenario_titles[scenario_title]["plot"]

    for jdx, method in enumerate(method_titles):
        if method == "data":
            data = read_and_stack_counts(scenario, bg, instrument)
        elif method == "gt":
            data = read_flux_ref(scenario)
        else:
            data = read_flux(scenario, bg, instrument)

        norm = simple_norm(data, stretch="asinh", min_cut=0, **norm_kwargs)

        ax = axes[idx, jdx]
        im = ax.imshow(data, cmap="viridis", origin="lower", norm=norm)

        if scenario == "point1":
            ax.set_title(method_titles[method], fontsize=9)

        ax.set_axis_off()

        if method == "data":
            ax.text(
                x=-40,
                y=DATA_SHAPE[0] / 2.0,
                s=scenario_title,
                fontsize=12,
                va="center",
            )

        if jdx == len(method_titles) - 1:
            add_cbar(im, ax, fig, label="Flux / A.U.")


plt.savefig(paths.figures / "comparison-scenarios.pdf", dpi=config.DPI)
