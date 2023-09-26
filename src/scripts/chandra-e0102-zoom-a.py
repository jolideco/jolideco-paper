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

filenames_profiles_counts = list(
    (PATH_RESULTS / "profiles").glob("*/e0102-zoom-a-iter-*-profile-counts.fits")
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


def measure_fwhm(d, spline_smoothing=0.0):
    from scipy.interpolate import UnivariateSpline

    x = d["x_normed"].to_value("arcsec")
    y = d["mean"]

    # create a spline of x and blue-np.max(blue)/2
    y_half = y - np.max(y) / 2
    spline = UnivariateSpline(x, y_half, s=spline_smoothing)
    r1, r2 = spline.roots()  # find the roots
    fwhm = r2 - r1
    return fwhm, r1, r2, np.max(y) / 2.0


def normalize_profile(profile, smooth_offset=0):
    """Normalize profile to integral"""
    profile_data = profile.profile

    x = profile.x_ref

    smoothed = gaussian_filter(profile_data, smooth_offset)
    offset = x[np.argmax(smoothed)]

    return {
        "x_normed": x - offset,
        "mean": profile_data,
    }


def plot_fwhm(ax, fwhm, y_max_half, color, subscript, va="top"):
    r1 = -fwhm / 2
    r2 = fwhm / 2

    ax.annotate(
        text="",
        xy=(r1, y_max_half),
        xytext=(r2, y_max_half),
        color=color,
        arrowprops=dict(arrowstyle="<->", color=color),
    )

    if va == "top":
        y = y_max_half * 0.98
    else:
        y = y_max_half * 1.01

    ax.text(
        (r1 + r2) / 2,
        y,
        f"$\lambda_{{{subscript}}}$",
        ha="center",
        va=va,
        color=color,
    )


def get_mean_and_std(profiles, smooth_offset=0):
    """Compute mean and std of boostrapped profiles"""
    profile_data = np.array([_.table["profile"].quantity for _ in profiles])

    x = profiles[0].x_ref
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


def plot_profile(ax, d, label, color, n_sigma=3, ls="-"):
    x = d["x_normed"].to_value("arcsec")
    ax.plot(x, d["mean"], label=label, color=color, ls=ls)
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

cutout.plot(
    ax_image,
    cmap="viridis",
    interpolation="gaussian",
    add_cbar=False,
)

artist = region.to_pixel(cutout.geom.wcs).as_artist(
    facecolor="none", edgecolor="white", lw=0.5
)
ax_image.add_artist(artist)


d_counts = get_mean_and_std(read_profiles(filenames_profiles_counts), smooth_offset=3.0)

profiles_counts = read_profiles(filenames_profiles_counts)

fwhms_counts = np.array(
    [
        measure_fwhm(normalize_profile(_), spline_smoothing=0.34e-3)[0]
        for _ in profiles_counts
    ]
)

ax.text(
    s=f"$\lambda_C={fwhms_counts.mean():.2f}\pm{fwhms_counts.std():.2f}$ arcsec",
    x=0.59,
    y=0.8,
    transform=ax.transAxes,
    ha="left",
    va="top",
    color=config.COLORS["viridis-2"],
)

y_max_half = d_counts["mean"].max() / 2.0
plot_fwhm(
    ax,
    fwhm=fwhms_counts.mean(),
    y_max_half=y_max_half,
    color=config.COLORS["viridis-2"],
    subscript="C",
)

plot_profile(
    ax,
    d_counts,
    label="Stacked counts profile",
    color=config.COLORS["viridis-2"],
)


profiles_jolideco = read_profiles(filenames_profiles_jolideco)
fwhms_jolideco = np.array(
    [measure_fwhm(normalize_profile(_))[0] for _ in profiles_jolideco]
)

ax.text(
    s=f"$\lambda_J={fwhms_jolideco.mean():.2f}\pm{fwhms_jolideco.std():.2f}$ arcsec",
    x=0.59,
    y=0.9,
    transform=ax.transAxes,
    ha="left",
    va="top",
    color=config.COLORS["viridis-0"],
)


d = get_mean_and_std(profiles_jolideco, smooth_offset=0)
y_max_half = d["mean"].max() / 2.0
plot_profile(ax, d, label="Jolideco profile", color=config.COLORS["viridis-0"], ls="--")
plot_fwhm(
    ax,
    fwhm=fwhms_jolideco.mean(),
    y_max_half=y_max_half,
    color=config.COLORS["viridis-0"],
    subscript="J",
    va="bottom",
)

ax.set_ylabel("Normalized Flux")
ax.set_xlabel("Offset / arcsec")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylim(0, 1.8e-2)
ax.set_xlim(-1.4, 1.4)
ax.legend(loc="upper left", frameon=False, fontsize=9)

plt.savefig(paths.figures / "chandra-e0102-zoom-a.pdf", facecolor="w", dpi=300)
