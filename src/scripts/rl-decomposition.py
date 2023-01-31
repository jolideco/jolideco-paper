import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from skimage.restoration import richardson_lucy
from astropy.visualization import simple_norm
from scipy.ndimage import gaussian_filter
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft
import paths
import config


def disk_source_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=3,
    source_level=1_000,
    source_radius=6,
    background_level=0,
    random_state=None,
):
    """Get disk source with Gaussian PSF test data.
    The exposure has a gradient of 50% from left to right.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    source_radius : float
        Radius of the disk source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape)

    flux = (
        source_level
        * Tophat2DKernel(
        radius=source_radius, x_size=shape[1], y_size=shape[1], mode="oversample"
    ).array
    )

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = np.clip(convolve_fft((flux + background) * exposure, psf), 0, np.inf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux,
    }


random_state = np.random.RandomState(936)

dataset = disk_source_gauss_psf(random_state=random_state)

figsize = config.FigureSizeAA(aspect_ratio=1., width_aa="two-column")

n_iters_rl = [1, 10, 100, 1000]

ncols = len(n_iters_rl) + 1

gridspec_kw = {
    "wspace": 0.1,
    "left": 0.01,
    "right": 0.9,
    "bottom": 0.1,
}

fig = plt.figure(figsize=figsize.inch)
gs = gridspec.GridSpec(nrows=4, ncols=1, bottom=0.1, hspace=0.5, top=1)

fig_top = fig.add_subfigure(gs[:3, 0])
fig_bottom = fig.add_subfigure(gs[3:, 0])

gs_bottom = gridspec.GridSpec(nrows=1, ncols=ncols, **gridspec_kw)
axes_bottom = fig_bottom.subplots(
    nrows=1, ncols=3, gridspec_kw={"left": 0.2}
)

axes = fig_top.subplots(
    nrows=3, ncols=ncols, gridspec_kw=gridspec_kw
)

axes[0, 0].axis("off")
axes[0, 0].text(
    x=0.5, y=0.5, s="Reconstruction", size=config.FONTSIZE, ha="center", va="center"
)

axes[1, 0].axis("off")
offset = 0.18
axes[1, 0].text(
    x=0.5,
    y=0.5 + offset,
    s="Reconstruction",
    size=config.FONTSIZE,
    ha="center",
    va="center",
)
axes[1, 0].text(
    x=0.5, y=0.5, s="$\circledast$", size=config.FONTSIZE, ha="center", va="center"
)
axes[1, 0].text(
    x=0.5, y=0.5 - offset, s="PSF", size=config.FONTSIZE, ha="center", va="center"
)

axes[2, 0].axis("off")
axes[2, 0].text(
    x=0.5,
    y=0.5 + offset / 2,
    s="Residual",
    size=config.FONTSIZE,
    ha="center",
    va="center",
)
axes[2, 0].text(
    x=0.5,
    y=0.5 - offset / 2,
    s="Counts",
    size=config.FONTSIZE,
    ha="center",
    va="center",
)

vmax = 8

counts = dataset["counts"]
psf = dataset["psf"]
exposure = dataset["exposure"]
background = dataset["background"]

axes_bottom[0].imshow(counts, origin="lower", vmin=0, vmax=vmax)
axes_bottom[0].axis("off")
axes_bottom[0].set_title("Counts", size=config.FONTSIZE)

axes_bottom[1].imshow(psf, origin="lower", vmin=0)
axes_bottom[1].axis("off")
axes_bottom[1].set_title("PSF", size=config.FONTSIZE)

axes_bottom[2].imshow(dataset["flux"], origin="lower", vmin=0, vmax=vmax)
axes_bottom[2].axis("off")
axes_bottom[2].set_title("Ground Truth", size=config.FONTSIZE)


def add_cbar(im, ax, fig):
    bbox = ax.get_position()
    loright = bbox.corners()[-2]
    rect = [loright[0] + 0.02, loright[1], 0.02, bbox.height]
    cax = fig.add_axes(rect)
    return fig.colorbar(im, cax=cax, orientation="vertical")


for idx, n_iter in enumerate(n_iters_rl):
    idx = idx + 1
    image = np.nan_to_num((counts / exposure)) - background
    flux_reco = richardson_lucy(
        np.clip(image, 0, np.inf), psf=psf, num_iter=n_iter, clip=False
    )
    flux_reco = flux_reco

    im = axes[0, idx].imshow(flux_reco, vmin=0, vmax=vmax)
    axes[0, idx].axis("off")
    axes[0, idx].set_title(f"$N_{{iter}} = {n_iter}$", size=config.FONTSIZE)

    if idx == 4:
        cbar = add_cbar(im=im, ax=axes[0, idx], fig=fig_top)

    flux_reco_psf = convolve_fft(flux_reco, psf)
    im = axes[1, idx].imshow(flux_reco_psf, vmin=0, vmax=vmax)
    axes[1, idx].axis("off")

    if idx == 4:
        cbar = add_cbar(im=im, ax=axes[1, idx], fig=fig_top)

    residual = gaussian_filter((image - flux_reco_psf), 1)
    im = axes[2, idx].imshow(residual, vmin=-4, vmax=4, cmap="RdBu")
    axes[2, idx].axis("off")

    if idx == 4:
        cbar = add_cbar(im=im, ax=axes[2, idx], fig=fig_top)

plt.savefig(paths.figures / "richardson-lucy-decomposition.pdf", facecolor="w", dpi=300)
