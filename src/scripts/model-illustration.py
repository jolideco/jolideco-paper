import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from matplotlib import patches
from matplotlib.transforms import Bbox
from scipy.special import gamma

random_state = np.random.RandomState(49837)

figsize = config.FigureSizeAA(aspect_ratio=1.9, width_aa="two-column")

fig = plt.figure(figsize=figsize.inch)

image_size = (128, 128)

GRAY = (0.7, 0.7, 0.7)


def draw_circular_arrow(ax, x, y, angle_start, radius=0.15, angle=90, **kwargs):
    kwargs.setdefault("color", GRAY)
    theta1 = 0
    theta2 = angle

    if angle_start < 180:
        width, height = 2 * radius, radius
        y_head = y + radius
    else:
        width, height = radius, 2 * radius
        y_head = y - radius

    arc = patches.Arc(
        xy=[x, y],
        width=width,
        height=height,
        angle=angle_start,
        theta1=theta1,
        theta2=theta2,
        linestyle="-",
        color=kwargs.get("color", "black"),
        lw=5,
    )
    ax.add_patch(arc)

    x_head = x + 0.01
    ax.scatter(x_head, y_head, marker=">", s=100, color=kwargs.get("color", "black"))


def shrink_axis(ax, factor=0.55, shift_x=0, shift_y=0):
    """Shrink axis by *factor*."""
    x0, y0, width, height = ax.get_position().bounds

    x0 += (1 - factor) / 2 * width + shift_x
    y0 += (1 - factor) / 2 * height + shift_y
    width *= factor
    height *= factor
    ax.set_position(Bbox.from_bounds(x0, y0, width, height))


def get_flux():
    """Get flux image from MSX data."""
    filename = get_pkg_data_filename("galactic_center/gc_msx_e.fits")
    return 1e5 * fits.getdata(filename)[10:138, 0:128]


def get_exposure():
    """Create exposure image with four patches."""
    exposure = np.zeros(image_size)
    centers = [(35, 37), (90, 38), (36, 90), (88, 92)]

    width = 60

    for idx, jdx in centers:
        slices_ = (
            slice(idx - width // 2, idx + width // 2),
            slice(jdx - width // 2, jdx + width // 2),
        )
        exposure[slices_] += 1.0

    return exposure


gridspec_kw = {
    "top": 0.92,
    "bottom": 0.02,
    "left": 0.01,
    "right": 0.99,
}

fig, axes = plt.subplots(
    nrows=2, ncols=4, figsize=figsize.inch, gridspec_kw=gridspec_kw
)

for ax in axes.flat:
    ax.set_axis_off()

flux = get_flux()
norm = simple_norm(flux, "asinh", asinh_a=0.005, min_cut=0)
axes[0, 0].imshow(flux, origin="lower", cmap="viridis", norm=norm)
axes[0, 0].set_title("Flux")


psf = Gaussian2DKernel(2.0)

shrink_axis(axes[1, 0])
axes[1, 0].imshow(psf, origin="lower", cmap="viridis", alpha=0.7)
axes[1, 0].set_title("PSF", alpha=0.7, size=10)


flux_psf = convolve_fft(flux, psf)
axes[1, 1].imshow(flux_psf, origin="lower", cmap="viridis", norm=norm)
axes[1, 1].set_title("Flux $\circledast$ PSF")

exposure = get_exposure()

shrink_axis(axes[0, 1], shift_x=-0.02)
axes[0, 1].imshow(exposure, origin="lower", cmap="viridis", alpha=0.7)
axes[0, 1].set_title("Exposure", alpha=0.7, size=10)


flux_psf_exposure = flux_psf * exposure
axes[0, 2].imshow(
    flux_psf_exposure,
    origin="lower",
    cmap="viridis",
    interpolation="nearest",
    norm=norm,
)
axes[0, 2].set_title("Flux $\circledast$ PSF $\cdot$ Exposure")

shrink_axis(axes[1, 2], shift_x=-0.02)
# using step size as 1


def poisson_pmf(x, lambda_):
    return (lambda_**x) * np.exp(-lambda_) / gamma(x + 1.0)


x = np.linspace(0, 10, 100)
y = poisson_pmf(x, lambda_=3)

axes[1, 2].plot(x, y, color="#400F59", lw=3, alpha=0.7)
axes[1, 2].set_title("Poisson Noise", alpha=0.7, size=10)
axes[1, 2].set_axis_on()
axes[1, 2].set_xticks([])
axes[1, 2].set_yticks([])
axes[1, 2].spines[["right", "top"]].set_visible(False)


counts = random_state.poisson(flux_psf_exposure)
axes[1, 3].imshow(
    counts, origin="lower", cmap="viridis", interpolation="nearest", norm=norm
)
axes[1, 3].set_title("Counts")

ax_fig = fig.add_axes([0, 0, 1, 1])

ax_fig.set_axis_off()

draw_circular_arrow(
    ax_fig,
    x=0.245,
    y=0.5,
    angle_start=180,
)
draw_circular_arrow(
    ax_fig,
    x=0.5,
    y=0.5,
    angle_start=90,
)
draw_circular_arrow(
    ax_fig,
    x=0.755,
    y=0.5,
    angle_start=180,
)

ax_fig.set_xlim(0, 1)
ax_fig.set_ylim(0, 1)

plt.savefig(paths.figures / "model-illustration.pdf", facecolor="w", dpi=300)
