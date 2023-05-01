import config
import numpy as np
import paths
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

asinh_a = 0.01

scenario_titles = {
    "Scenario A": {
        "name": "point1",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
    "Scenario B": {
        "name": "aster3",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
    "Scenario C": {
        "name": "disk3",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
    "Scenario D": {
        "name": "spiral4",
        "plot": {"max_cut": 200.0, "asinh_a": asinh_a},
    },
}


instrument_titles = {
    "ground truth": "Ground\n  Truth",
    "chandra": '"Chandra"',
    "xmm": '"XMM"',
}

figsize = config.FigureSizeAA(aspect_ratio=1.32, width_aa="two-column")

upsampling_factor = 2
DATA_SHAPE = (128, 128)

gridspec_kw = {
    "left": 0.14,
    "right": 0.98,
    "bottom": 0.12,
    "top": 0.95,
    "wspace": 0.05,
    "hspace": 0.02,
}

fig, axes = plt.subplots(
    nrows=len(instrument_titles),
    ncols=len(scenario_titles),
    figsize=figsize.inch,
    gridspec_kw=gridspec_kw,
)


def add_cbar(im, ax, fig, label=""):
    """Add cbar to a given axis and figure"""
    bbox = ax.get_position()
    loleft = bbox.corners()[0]
    rect = [loleft[0], loleft[1] - 0.04, bbox.width, 0.02]
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label, size=9)
    cbar.set_ticks([0, 10, 100])
    cax.tick_params(labelsize=9)


def read_flux_ref(instrument, scenario):
    """Reda reference flux"""
    path = paths.jolideco_repo_comparison / "data"

    if instrument == "ground truth":
        pattern = f"chandra_gauss_fwhm4710_128x128_mod*_{scenario}.fits"
    elif instrument == "chandra":
        pattern = f"chandra_gauss_fwhm4710_128x128_img*_{scenario}.fits"
    else:
        pattern = f"xmm_gauss_fwhm14130_128x128_img*_{scenario}.fits"

    filenames = path.glob(pattern)
    return fits.getdata(list(filenames)[0])


for idx, scenario_title in enumerate(scenario_titles):
    scenario = scenario_titles[scenario_title]["name"]
    norm_kwargs = scenario_titles[scenario_title]["plot"]

    for jdx, instrument in enumerate(instrument_titles):
        data = read_flux_ref(instrument, scenario)

        norm = simple_norm(data, stretch="asinh", min_cut=0, **norm_kwargs)

        ax = axes[jdx, idx]
        im = ax.imshow(data, cmap="viridis", origin="lower", norm=norm)

        if instrument == "ground truth":
            ax.set_title(scenario_title, fontsize=12)

        ax.set_axis_off()

        if scenario == "point1":
            ax.text(
                x=-45,
                y=DATA_SHAPE[0] / 2.0,
                s=instrument_titles[instrument],
                fontsize=12,
                va="center",
                ha="center",
            )

        if instrument == "xmm":
            add_cbar(im, ax, fig, label="Expected Counts")


plt.savefig(paths.figures / "scenarios.pdf", dpi=config.DPI)
