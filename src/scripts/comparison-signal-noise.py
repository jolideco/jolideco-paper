import config
import matplotlib.lines as lines
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

bg_titles = [
    ("bg1", "Data"),
    ("bg1", "Reconstruction"),
    ("bg2", "Data"),
    ("bg2", "Reconstruction"),
    ("bg3", "Data"),
    ("bg3", "Reconstruction"),
]

figsize = config.FigureSizeAA(aspect_ratio=1.2, width_aa="two-column")

upsampling_factor = 2
DATA_SHAPE = (128, 128)

gridspec_kw = {
    "left": 0.05,
    "right": 0.99,
    "bottom": 0.01,
    "top": 0.91,
    "wspace": 0.05,
    "hspace": 0.05,
}

fig, axes = plt.subplots(
    nrows=len(scenario_titles),
    ncols=len(bg_titles),
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


def move_axis(ax):
    """Move axis"""
    box = ax.get_position()
    box.x0 = box.x0 - 0.007
    box.x1 = box.x1 - 0.007
    ax.set_position(box)


for idx, scenario_title in enumerate(scenario_titles):
    scenario = scenario_titles[scenario_title]["name"]
    norm_kwargs = scenario_titles[scenario_title]["plot"]

    for jdx, (bg, bg_title) in enumerate(bg_titles):
        if bg_title == "Data":
            data = read_and_stack_counts(scenario, bg, instrument)
        elif scenario == "gt":
            data = read_flux_ref(scenario)
        else:
            data = read_flux(scenario, bg, instrument)

        norm_kwargs["max_cut"] = np.percentile(data, 99.9)
        norm = simple_norm(data, stretch="asinh", min_cut=0, **norm_kwargs)
        ax = axes[idx, jdx]
        ax.imshow(data, cmap="viridis", origin="lower", norm=norm)

        if scenario == "spiral1":
            ax.set_title(bg_title, fontsize=9)

        ax.set_axis_off()

        if jdx == 0:
            ax.text(
                x=-37,
                y=DATA_SHAPE[0] // 2.0,
                s=scenario_title,
                fontsize=12,
                va="center",
            )

        if (jdx + 1) % 2 == 0:
            move_axis(ax)


x_1, x_2 = 0.358, 0.674
y_1, y_2 = 0.01, 0.96
color = 3 * (0.5,)
fig.add_artist(lines.Line2D([x_1, x_1], [y_1, y_2], color=color, lw=0.75))
fig.add_artist(lines.Line2D([x_2, x_2], [y_1, y_2], color=color, lw=0.75))

y = 0.97
x_diff = 0.315
x_0 = 0.21
fig.text(x_0, y=y, s="Bkg 1", fontsize=12, ha="center", va="center")
fig.text(x_0 + x_diff, y=y, s="Bkg 2", fontsize=12, ha="center", va="center")
fig.text(x_0 + 2 * x_diff, y=y, s="Bkg 3", fontsize=12, ha="center", va="center")

plt.savefig(paths.figures / "comparison-signal-noise.pdf", dpi=config.DPI)
