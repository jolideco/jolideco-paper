import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import simple_norm
from gammapy.estimators.utils import find_peaks
from gammapy.maps import Map
from regions import RectangleSkyRegion

SMOOTH_WIDTH = 5
FONT_SIZE = 8

plt.rcParams.update({"font.size": FONT_SIZE})


def format_axes(ax, hide_yaxis=False):
    """Format axes for a map plot"""
    lon = ax.coords["ra"]
    lat = ax.coords["dec"]

    lon.set_ticks_position("b")
    lon.set_ticklabel_position("b")

    lat.set_ticks_position("l")
    lat.set_ticklabel_position("l")

    lon.set_major_formatter("d.dd")
    lat.set_major_formatter("d.ddd")

    lon.set_ticks(spacing=1 * u.arcmin)
    lat.set_ticks(spacing=10 * u.arcsec)

    if hide_yaxis:
        lat.set_axislabel("")
        lat.set_ticklabel_visible(False)


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


def draw_zoom_lines(ax, ax_zoom):
    """Draw zoom lines from one axis to another"""
    x = [0.05, 0.05]
    y = [0.05, 0.05]
    x2 = [0.95, 0.95]
    y2 = [0.95, 0.95]

    (x1a, y1a), (x2a, y2a) = ax.transAxes.transform([(0, 0), (1, 1)])
    (x1b, y1b), (x2b, y2b) = ax_zoom.transAxes.transform([(0, 0), (1, 1)])

    ax_zoom.plot(
        [x1a, x1b], [y1a, y1b], color="white", transform=fig.transFigure, clip_on=False
    )
    ax_zoom.plot(
        [x2a, x2b], [y2a, y2b], color="white", transform=fig.transFigure, clip_on=False
    )


def draw_zoom_box(ax, center, width):
    """Draw zoom box"""
    region = RectangleSkyRegion(center=center, width=width, height=width)
    region_pix = region.to_pixel(wcs=ax.wcs)
    ax.add_patch(region_pix.as_artist(facecolor="none", edgecolor="white", lw=0.5))


path = paths.jolideco_repo_chandra_example / "results/e0102-broadband/jolideco"

filename_jolideco = path / "e0102-broadband-result-jolideco.fits"
filename_npred = path / "e0102-broadband-npred.fits"

filenames_data = path.parent.glob("*/maps/e0102-broadband-*-counts.fits")

npred = Map.read(filename_npred)

aspect_ratio = 1.8
figsize = config.FigureSizeAA(aspect_ratio=aspect_ratio, width_aa="two-column")

fig = plt.figure(figsize=figsize.inch)

wcs = npred.geom.wcs
height = 0.57
width = height / aspect_ratio
y_bottom = 0.1

ax_counts = fig.add_axes([0.11, y_bottom, width, height], projection=wcs)
format_axes(ax_counts)

ax_flux = fig.add_axes(
    [0.57, y_bottom, width, height], projection=npred.geom.upsample(2).wcs
)
format_axes(ax_flux, hide_yaxis=True)


stacked = Map.from_geom(geom=npred.geom)

for filename in filenames_data:
    counts = fits.getdata(filename)
    stacked += counts

print(f"Total counts: {stacked.data.sum()}")

norm_counts = simple_norm(
    stacked.data,
    stretch="linear",
    min_cut=0,
    max_cut=145,
)
stacked.plot(ax=ax_counts, cmap="viridis", interpolation="None", norm=norm_counts)
add_cbar(ax_counts.images[0], ax_counts, fig, label="Counts")

flux_data = fits.getdata(filename_jolideco, hdu="VELA-JUNIOR")
flux = Map.from_geom(npred.geom.upsample(2), data=flux_data)
norm_flux = simple_norm(
    flux.data,
    stretch="linear",
    min_cut=0,
    max_cut=3.5,
)
flux.plot(ax=ax_flux, cmap="viridis", interpolation="gaussian", norm=norm_flux)
add_cbar(ax_flux.images[0], ax_flux, fig, label="Flux / A.U.")

norm_factor = np.pi * SMOOTH_WIDTH**2
diff = (stacked - npred).smooth(SMOOTH_WIDTH) * norm_factor

residuals = diff / np.sqrt(npred.smooth(SMOOTH_WIDTH) * norm_factor)

residuals_per_pix = (stacked - npred) / np.sqrt(npred)
print(f"Mean residuals : {residuals_per_pix.data.mean():.3f}")
print(f"Sigma residuals: {residuals_per_pix.data.std():.3f}")


def draw_zoom(
    fig,
    ax,
    ax_zoom_rect,
    map_,
    center,
    width,
    title,
    norm=None,
):
    """Draw zoom box"""
    cutout = map_.cutout(position=center, width=width)

    ax_zoom = fig.add_axes(ax_zoom_rect, projection=cutout.geom.wcs)

    cutout.plot(ax=ax_zoom, cmap="viridis", interpolation="None", norm=norm)
    lon = ax_zoom.coords["ra"]
    lat = ax_zoom.coords["dec"]

    lat.set_axislabel("")
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)

    lon.set_axislabel("")
    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    draw_zoom_box(ax, center=center, width=width)
    ax_zoom.set_title(f"Zoom {title}", pad=3)

    y = center.icrs.dec.deg - width.to_value("deg") / 2.0
    y -= (1.0 * u.arcsec).to_value("deg")
    va = "top"

    ax.text(
        x=center.icrs.ra.deg,
        y=y,
        s=title,
        transform=ax.get_transform("icrs"),
        ha="center",
        va=va,
        color="white",
        size=8,
    )


# inset 1
center = SkyCoord("16.017d", "-72.034d")
width = 8 * u.arcsec

insets_bottom = 0.69
insets_width = 0.26
insets_counts_left = -0.04
insets_flux_left = 0.46
spacing = 0.16

draw_zoom(
    fig,
    ax=ax_counts,
    ax_zoom_rect=[insets_counts_left, insets_bottom, insets_width, insets_width],
    map_=stacked,
    center=center,
    width=width,
    title="A",
    norm=norm_counts,
)
draw_zoom(
    fig,
    ax=ax_flux,
    ax_zoom_rect=[insets_flux_left, insets_bottom, insets_width, insets_width],
    map_=flux,
    center=center,
    width=width,
    title="A",
    norm=norm_flux,
)

# inset 2
center = SkyCoord("16.01148d", "-72.03335d")
width = 3 * u.arcsec
draw_zoom(
    fig,
    ax=ax_counts,
    ax_zoom_rect=[
        insets_counts_left + spacing,
        insets_bottom,
        insets_width,
        insets_width,
    ],
    map_=stacked,
    center=center,
    width=width,
    title="B",
    norm=norm_counts,
)
draw_zoom(
    fig,
    ax=ax_flux,
    ax_zoom_rect=[
        insets_flux_left + spacing,
        insets_bottom,
        insets_width,
        insets_width,
    ],
    map_=flux,
    center=center,
    width=width,
    title="B",
    norm=norm_flux,
)

cutout_zoom_b = flux.cutout(position=center, width=width)

peaks = find_peaks(cutout_zoom_b, threshold=0.4)
position = SkyCoord(peaks["ra"], peaks["dec"], unit="deg", frame="icrs")

print(position.to_string("hmsdms"))


# inset 3
center = SkyCoord("16.003d", "-72.035d")
width = 8 * u.arcsec
draw_zoom(
    fig,
    ax=ax_counts,
    ax_zoom_rect=[
        insets_counts_left + 2 * spacing,
        insets_bottom,
        insets_width,
        insets_width,
    ],
    map_=stacked,
    center=center,
    width=width,
    title="C",
    norm=norm_counts,
)
draw_zoom(
    fig,
    ax=ax_flux,
    ax_zoom_rect=[
        insets_flux_left + 2 * spacing,
        insets_bottom,
        insets_width,
        insets_width,
    ],
    map_=flux,
    center=center,
    width=width,
    title="C",
    norm=norm_flux,
)

# norm = simple_norm(residuals.data, stretch="linear", min_cut=-2, max_cut=2)
# residuals.plot(ax=ax_residuals, cmap="RdBu", norm=norm, interpolation="gaussian")
# add_cbar(
#     ax_residuals.images[0],
#     ax_residuals,
#     fig,
#     label="$(N_{Counts} - N_{Pred}) / \sqrt{N_{Pred}}$",
# )

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

# ticks = np.round(norm.inverse(np.linspace(0, 1, 10)), 1)
# plt.colorbar(ax_npred.images[-1], cax=ax_cbar, ticks=ticks)

plt.savefig(paths.figures / "example-chandra.pdf", dpi=config.DPI)
