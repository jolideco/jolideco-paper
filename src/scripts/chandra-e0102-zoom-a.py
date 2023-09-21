from pathlib import Path

import config
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import simple_norm
from gammapy.estimators import ImageProfile
from gammapy.maps import Map, WcsGeom
from regions import RectangleSkyRegion
from scipy.ndimage import gaussian_filter

PATH_RESULTS = paths.jolideco_repo_chandra_e0102_zoom_a / "results" / "e0102-zoom-a"

center = SkyCoord(16.0172, -72.0340, unit="deg")

region = RectangleSkyRegion(
    center=center,
    width=5 * u.arcsec,
    height=0.6 * u.arcsec,
    angle=45 * u.deg,
)

GEOM_PROFILE = WcsGeom.create(
    skydir=region.center,
    width=(region.height, region.width),
    binsz=0.02 * u.arcsec,
    frame="galactic",
)

GEOM_PROFILE.wcs.wcs.crota = [45, 45]

filenames_profiles_jolideco = (PATH_RESULTS / "profiles").glob(
    "*/e0102-zoom-a-iter-*-profile-jolideco.fits"
)

filenames_profiles_counts = (PATH_RESULTS / "profiles").glob(
    "*/e0102-zoom-a-iter-*-profile-counts.fits"
)


def read_profiles(filenames):
    """Read profiles"""
    profiles = []

    for filename in filenames:
        table = Table.read(filename)
        profile = ImageProfile(table=table)
        profile = profile.normalize("integral")
        profiles.append(profile)

    return profiles


def measure_fwhm(d):
    from scipy.interpolate import UnivariateSpline

    x = d["x_normed"].quantity.to_value("arcsec")
    y = d["mean"]

    # create a spline of x and blue-np.max(blue)/2
    y_half = y - np.max(y) / 2
    spline = UnivariateSpline(x, y_half, s=0)
    r1, r2 = spline.roots()  # find the roots
    return r1, r2, np.max(y) / 2.0


def plot_fwhm(ax, d, color):
    r1, r2, y_max_half = measure_fwhm(d)

    ax.annotate(
        text="",
        xy=(r1, y_max_half),
        xytext=(r2, y_max_half),
        color=color,
        arrowprops=dict(arrowstyle="<->", color=color),
    )
    ax.text(
        (r1 + r2) / 2,
        y_max_half * 0.98,
        f"$\lambda_{{fwhm}}$={r2 - r1:.2f}",
        ha="center",
        va="top",
        color=color,
    )


def get_mean_and_std(profiles, smooth_offset=0):
    """Compute mean and std of boostrapped profiles"""
    profile_data = np.array([_.table["profile"].quantity for _ in profiles])

    x = profiles[0].table["x_ref"]
    mean = profile_data.mean(axis=0)
    std = profile_data.std(axis=0)

    smoothed = gaussian_filter(mean, smooth_offset)
    offset = x[np.argmax(smoothed)]

    return {
        "x_normed": x - offset,
        "x": x,
        "offset": offset,
        "mean": mean,
        "std": std,
    }


def plot_profile(ax, d, label, color, n_sigma=3):
    x = d["x_normed"].quantity.to_value("arcsec")
    ax.plot(x, d["mean"], label=label, color=color)
    ax.fill_between(
        x,
        d["mean"] - n_sigma * d["std"],
        d["mean"] + n_sigma * d["std"],
        alpha=0.2,
        color=color,
        linewidth=0.0,
    )


figsize = config.FigureSizeAA(aspect_ratio=2.5, width_aa="two-column")

fig = plt.figure(figsize=figsize.inch)

y_low = 0.15
ax = fig.add_axes([0.09, y_low, 0.57, 0.79])

ratio = figsize.aspect_ratio
width = 0.316
ax_image = fig.add_axes([0.67, y_low, width, ratio * width])
ax_image.set_xticks([])
ax_image.set_yticks([])
ax_image.text(s="Profile Region", x=35, y=55, color="white", fontsize=10, rotation=45)
ax_image.set_xlabel("Zoom A", fontsize=12)


path = paths.jolideco_repo_chandra_example / "results/e0102-broadband/jolideco"
filename_jolideco = path / "e0102-broadband-result-jolideco.fits"

filename_npred = path / "e0102-broadband-npred.fits"
filenames_data = path.parent.glob("*/maps/e0102-broadband-*-counts.fits")

npred = Map.read(filename_npred)
flux_data = fits.getdata(filename_jolideco, hdu="VELA-JUNIOR")
flux = Map.from_geom(npred.geom.upsample(2), data=flux_data.astype(float))

center = SkyCoord("16.017d", "-72.034d")

cutout = flux.cutout(position=center, width=8 * u.arcsec)

norm = simple_norm(cutout.data, stretch="asinh", asinh_a=0.5)
cutout.plot(
    ax_image, cmap="viridis", interpolation="gaussian", add_cbar=False, norm=norm
)

artist = region.to_pixel(cutout.geom.wcs).as_artist(
    facecolor="none", edgecolor="white", lw=0.5
)
ax_image.add_artist(artist)


# ax_image_profile = fig.add_axes([0.7, 0.12, width, 0.1])
# flux_stripe = flux.interp_to_geom(GEOM_PROFILE)
# norm = simple_norm(flux_stripe.data, stretch="linear", min_cut=0, max_cut=3.5)
# # ax_image_profile.set_xlabel("Offset (arcsec)")
# # ax_image_profile.set_yticks([])
# # ax_image_profile.set_xticks([-1, 0, 1])


# ax_image_profile.imshow(flux_stripe.data.T, cmap="viridis", norm=norm)


# fig, axes = plt.subplots(
#     nrows=1, ncols=1, figsize=figsize.inch, gridspec_kw={"left": 0.01, "right": 0.99}
# )


d_counts = get_mean_and_std(read_profiles(filenames_profiles_counts), smooth_offset=3.0)
plot_profile(
    ax, d_counts, label="Stacked counts profile", color=config.COLORS["viridis-2"]
)
# plot_fwhm(ax, d_counts, color=config.COLORS["viridis-2"])


d = get_mean_and_std(read_profiles(filenames_profiles_jolideco), smooth_offset=0)
plot_profile(ax, d, label="Jolideco profile", color=config.COLORS["viridis-0"])
plot_fwhm(ax, d, color=config.COLORS["viridis-0"])

# ax.plot(d["x"] - offset, smoothed)

ax.set_ylabel("Flux / A.U.")
ax.set_xlabel("Offset (arcsec)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
# ax.set_ylim(0, 1.2)
ax.set_xlim(-1.8, 1.8)
ax.legend(loc="upper left", frameon=False, fontsize=9)

plt.savefig(paths.figures / "chandra-e0102-zoom-a.pdf", facecolor="w", dpi=300)
