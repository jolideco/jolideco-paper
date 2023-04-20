import config
import numpy as np
import paths
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

scenario_titles = {
    "D1": {
        "name": "spiral1",
        "plot": {"max_cut": 10.0, "asinh_a": 0.02},
    },
    "D2": {
        "name": "spiral2",
        "plot": {"max_cut": 30.0, "asinh_a": 0.02},
    },
    "D3": {
        "name": "spiral3",
        "plot": {"max_cut": 30.0, "asinh_a": 0.02},
    },
    "D4": {
        "name": "spiral4",
        "plot": {"max_cut": 100.0, "asinh_a": 0.02},
    },
    "D5": {
        "name": "spiral5",
        "plot": {"max_cut": 5000.0, "asinh_a": 0.02},
    },
}

instrument = "chandra"
method = "jolideco-patch-prior-gleam-v0.1"

bg_titles = {
    "bg1": "Bkg 1",
    "bg2": "Bkg 2",
    "bg3": "Bkg 3",
}

figsize = config.FigureSizeAA(aspect_ratio=1.618, width_aa="two-column")

upsampling_factor = 2
DATA_SHAPE = (128, 128)

gridspec_kw = {
    "left": 0.08,
    "right": 0.99,
    "bottom": 0.02,
    "top": 0.92,
    "wspace": 0.05,
    "hspace": 0.05,
}

fig, axes = plt.subplots(
    nrows=len(bg_titles),
    ncols=len(scenario_titles),
    figsize=figsize.inch,
    gridspec_kw=gridspec_kw,
)


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

    for jdx, bg in enumerate(bg_titles):
        if scenario == "data":
            data = read_and_stack_counts(scenario, bg, instrument)
        elif scenario == "gt":
            data = read_flux_ref(scenario)
        else:
            data = read_flux(scenario, bg, instrument)

        norm_kwargs["max_cut"] = np.percentile(data, 99.9)
        norm = simple_norm(data, stretch="asinh", min_cut=0, **norm_kwargs)
        ax = axes[jdx, idx]
        ax.imshow(data, cmap="viridis", origin="lower", norm=norm)

        if bg == "bg1":
            ax.set_title(scenario_title, fontsize=12)

        ax.set_axis_off()

        if scenario == "spiral1":
            ax.text(
                x=-100,
                y=DATA_SHAPE[0],
                s=bg_titles[bg],
                fontsize=12,
                va="center",
            )


plt.savefig(paths.figures / "comparison-singal-noise.pdf", dpi=config.DPI)
