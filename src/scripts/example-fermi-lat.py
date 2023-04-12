import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from gammapy.maps import Map

SMOOTH_WIDTH = 5

path = (
    paths.jolideco_repo_fermi_lat_example
    / "results/vela-junior-above-10GeV-data/jolideco"
)

filename_jolideco = path / "vela-junior-above-10GeV-data-result-jolideco.fits"
filenames_data = (path / "input").glob("*.fits")
filename_npred = path / "vela-junior-above-10GeV-data-result-npred.fits"


npred = Map.read(filename_npred)

figsize = config.FigureSizeAA(aspect_ratio=1.618, width_aa="two-column")

fig, ax = plt.subplots(figsize=figsize.inch)

wcs = npred.geom.wcs
height = 0.8
ax_counts = fig.add_axes([0.01, 0.12, 0.4, height], projection=wcs)
ax_flux = fig.add_axes([0.28, 0.12, 0.4, height], projection=wcs)
ax_residuals = fig.add_axes([0.55, 0.12, 0.4, height], projection=wcs)
ax_cbar = fig.add_axes([0.9, 0.12, 0.02, height])

ax_counts.set_title("Stacked Counts")
ax_flux.set_title("Reconstruction Jolideco")
ax_residuals.set_title("Stacked Residuals")


stacked = Map.from_geom(geom=npred.geom)

for filename in filenames_data:
    counts = Map.read(filename, hdu="COUNTS")
    stacked.stack(counts)


stacked.plot(ax=ax_counts, cmap="viridis", interpolation="none")

flux = Map.read(filename_jolideco, hdu="FLUX")
flux.plot(ax=ax_flux, cmap="viridis", interpolation="gaussian")


residuals = (stacked.smooth(SMOOTH_WIDTH) - npred.smooth(SMOOTH_WIDTH)) / np.sqrt(
    npred.smooth(SMOOTH_WIDTH)
)

residuals.plot(ax=ax_residuals, cmap="RdBu", vmin=-0.5, vmax=0.5)

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

ax_cbar.set_ylabel("Counts")

plt.savefig(paths.figures / "example-fermi-lat.pdf", dpi=config.DPI)
