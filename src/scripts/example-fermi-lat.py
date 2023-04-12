import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy import units as u
from astropy.io import fits
from gammapy.maps import Map

SMOOTH_WIDTH = 5
FONT_SIZE = 8

plt.rcParams.update({"font.size": FONT_SIZE})


def format_axes(ax, hide_yaxis=False):
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


def add_cbar(im, ax, fig, label=""):
    """Add cbar to a given axis and figure"""
    bbox = ax.get_position()
    loright = bbox.corners()[-2]
    rect = [loright[0] + 0.01, loright[1], 0.02, bbox.height]
    cax = fig.add_axes(rect)
    cax.set_ylabel(label)
    return fig.colorbar(im, cax=cax, orientation="vertical")


path = (
    paths.jolideco_repo_fermi_lat_example
    / "results/vela-junior-above-10GeV-data/jolideco"
)

filename_jolideco = path / "vela-junior-above-10GeV-data-result-jolideco.fits"
filenames_data = (path / "input").glob("*.fits")
filename_npred = path / "vela-junior-above-10GeV-data-npred.fits"

npred = Map.read(filename_npred)

aspect_ratio = 3.8
figsize = config.FigureSizeAA(aspect_ratio=aspect_ratio, width_aa="two-column")

fig = plt.figure(figsize=figsize.inch)

wcs = npred.geom.wcs
height = 0.75
width = height / aspect_ratio
y_bottom = 0.22

ax_counts = fig.add_axes([0.08, y_bottom, width, height], projection=wcs)
format_axes(ax_counts)

ax_flux = fig.add_axes([0.4, y_bottom, width, height], projection=wcs)
format_axes(ax_flux, hide_yaxis=True)

ax_residuals = fig.add_axes([0.7, y_bottom, width, height], projection=wcs)
format_axes(ax_residuals, hide_yaxis=True)


stacked = Map.from_geom(geom=npred.geom)

for filename in filenames_data:
    counts = Map.read(filename, hdu="COUNTS").sum_over_axes(keepdims=False)
    stacked.stack(counts)

stacked.plot(ax=ax_counts, cmap="viridis", interpolation="None")
add_cbar(ax_counts.images[0], ax_counts, fig, label="Counts")

flux_data = fits.getdata(filename_jolideco, hdu="VELA-JUNIOR")
flux = Map.from_geom(npred.geom, data=flux_data)
flux.plot(ax=ax_flux, cmap="viridis", interpolation="gaussian")
add_cbar(ax_flux.images[0], ax_flux, fig, label="Flux / $(10^{-14} cm^{-2} s^{-1})$")


residuals = (stacked.smooth(SMOOTH_WIDTH) - npred.smooth(SMOOTH_WIDTH)) / np.sqrt(
    npred.smooth(SMOOTH_WIDTH)
)

residuals.plot(ax=ax_residuals, cmap="RdBu", vmin=-0.5, vmax=0.5)
add_cbar(
    ax_residuals.images[0],
    ax_residuals,
    fig,
    label="(Counts - $N_{Pred}$) / $\sqrt{N_{Pred}}$",
)

# crop = slice(None)
# psf_data = dataset_4683["psf"][crop, crop] * norm.vmax
# size = psf_data.shape[0] * height / counts.data.shape[0]

# ax_psf = fig.add_axes([0.05, 0.7, size, size])
# ax_psf.imshow(psf_data, cmap=cmap, interpolation="gaussian", norm=norm)
# ax_psf.set_title("PSF", color="white", fontweight="bold")
# ax_psf.set_xticks([])
# ax_psf.set_yticks([])

# for spine in ax_psf.spines.values():
#     spine.set_edgecolor("white")
#     spine.set_lw(1.2)

# counts.plot(
#     ax=ax_counts,
#     norm=norm,
#     cmap=cmap,
#     interpolation="None",
# )
# npred.plot(ax=ax_npred, norm=norm, interpolation="gaussian", cmap=cmap)
# npred_rl.plot(ax=ax_npred_rl, norm=norm, interpolation="gaussian", cmap=cmap)

# lon = ax_npred.coords["ra"]
# lat = ax_npred.coords["dec"]
# lat.set_ticklabel_visible(False)

# lon = ax_npred_rl.coords["ra"]
# lat = ax_npred_rl.coords["dec"]
# lat.set_ticklabel_visible(False)

# ticks = np.round(norm.inverse(np.linspace(0, 1, 10)), 1)
# plt.colorbar(ax_npred.images[-1], cax=ax_cbar, ticks=ticks)

plt.savefig(paths.figures / "example-fermi-lat.pdf", dpi=config.DPI)
