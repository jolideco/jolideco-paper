import config
import numpy as np
import paths
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

scenario_titles = {
    "A": {
        "name": "point1",
        "plot": {"max_cut": 340.0, "asinh_a": 0.02},
    },
    "B": {
        "name": "aster3",
        "plot": {"max_cut": 200.0, "asinh_a": 0.02},
    },
    "C": {
        "name": "disk3",
        "plot": {"max_cut": 200.0, "asinh_a": 0.02},
    },
    "D": {
        "name": "spiral3",
        "plot": {"max_cut": 100.0, "asinh_a": 0.02},
    },
}

bg = "bg1"
instrument = "chandra"


method_titles = {
    "data": "Data",
    "gt": "Ground Truth",
    "pylira": "Pylira",
    "jolideco-uniform-prior=n=10": "Jolideco\n(Uni, n=10)",
    "jolideco-uniform-prior=n=1000": "Jolideco\n(Unif., n=1000)",
    "jolideco-patch-prior-gleam-v0.1": "Jolideco\n(GLEAM v0.1)",
    "jolideco-patch-prior-zoran-weiss": "Jolideco\n(Zoran-Weiss)",
}

figsize = config.FigureSizeAA(aspect_ratio=1.618, width_aa="two-column")

upsampling_factor = 2
DATA_SHAPE = (128, 128)

gridspec_kw = {
    "left": 0.05,
    "right": 0.98,
    "bottom": 0.02,
    "top": 0.92,
    "wspace": 0.05,
    "hspace": 0.05,
}

fig, axes = plt.subplots(
    nrows=len(scenario_titles),
    ncols=len(method_titles),
    figsize=figsize.inch,
    gridspec_kw=gridspec_kw,
)


def read_and_stack_counts(scenario, bg, instrument):
    """Read and stack couns files"""
    path = paths.jolideco_repo_comparison / "data"

    filenames = path.glob(
        f"{instrument}_gauss_fwhm4710_128x128_sim*_{bg}_{scenario}_iter*.fits"
    )

    data = np.zeros(DATA_SHAPE)

    for filename in filenames:
        data += fits.getdata(filename)

    return data


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
        axes[idx, jdx].imshow(data, cmap="viridis", origin="lower", norm=norm)

        if scenario == "point1":
            axes[idx, jdx].set_title(method_titles[method], fontsize=9)

        axes[idx, jdx].set_axis_off()

        if method == "data":
            axes[idx, jdx].text(
                x=-30,
                y=DATA_SHAPE[0] / 2.0,
                s=scenario_title,
                fontsize=12,
                va="center",
            )


plt.savefig(paths.figures / "comparison-scenarios.pdf", dpi=config.DPI)
