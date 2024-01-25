import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy import units as u
from astropy.visualization import simple_norm
from gammapy.maps import Map
from matplotlib.gridspec import GridSpec

SMOOTH_WIDTH = 5
FONT_SIZE = 8

plt.rcParams.update({"font.size": FONT_SIZE})


def format_axes(ax, hide_yaxis=False, hide_xaxis=False):
    """Format axes for a map plot"""
    lon = ax.coords["glon"]
    lat = ax.coords["glat"]

    lon.set_ticks_position("b")
    lon.set_ticklabel_position("b")

    lat.set_ticks_position("l")
    lat.set_ticklabel_position("l")

    lon.set_major_formatter("d.d")
    lat.set_major_formatter("d.d")

    lon.set_ticks(spacing=1.0 * u.deg)
    lat.set_ticks(spacing=0.5 * u.deg)

    if hide_yaxis:
        lat.set_axislabel("")
        lat.set_ticklabel_visible(False)

    if hide_xaxis:
        lon.set_axislabel("")
        lon.set_ticklabel_visible(False)


def add_cbar(im, ax, fig, label=""):
    """Add cbar to a given axis and figure"""
    bbox = ax.get_position()
    loright = bbox.corners()[-2]
    rect = [loright[0] + 0.005, loright[1], 0.02, bbox.height]
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(label)


def smooth_map(smooth_width, map):
    """Return a smoothed map"""
    return map.smooth(width=smooth_width) * smooth_width**2


path = (
    paths.jolideco_repo_fermi_lat_example
    / "results/vela-junior-above-10GeV-data/jolideco"
)


filenames = path.glob("vela-junior-above-10GeV-residual-psf*.fits")
filename_calibrated = path.glob("vela-junior-above-10GeV-residual-calibrated-psf*.fits")


aspect_ratio = 2.0
figsize = config.FigureSizeAA(aspect_ratio=aspect_ratio, width_aa="two-column")

fig = plt.figure(figsize=figsize.inch)

grid = GridSpec(
    2,
    4,
    figure=fig,
    left=0.08,
    right=0.9,
    bottom=0.12,
    top=0.95,
    wspace=0.1,
    hspace=0.1,
)

norm_factor = np.pi * SMOOTH_WIDTH**2

for jdx, filenames in enumerate([filenames, filename_calibrated]):
    for idx, filename in enumerate(filenames):
        residuals = Map.read(filename) * np.sqrt(norm_factor)
        ax = fig.add_subplot(grid[jdx, idx], projection=residuals.geom.wcs)
        norm = simple_norm(residuals.data, stretch="linear", min_cut=-2, max_cut=2)
        residuals.plot(ax=ax, cmap="RdBu", norm=norm, interpolation="gaussian")

        if jdx == 0:
            ax.set_title(f"PSF {idx + 1}", pad=4)

        format_axes(ax, hide_yaxis=idx != 0, hide_xaxis=jdx == 0)

        if idx == 3:
            im = ax.images[-1]
            add_cbar(
                im,
                ax,
                fig,
                label="$(N_{Counts} - N_{Pred}) / \sqrt{N_{Pred}}$",
            )


plt.savefig(paths.figures / "example-fermi-lat-non-calibrated.pdf", dpi=config.DPI)
