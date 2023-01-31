import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS

figsize = config.FigureSizeAA(aspect_ratio=13 / 4.0, width_aa="two-column")


fig = plt.figure(figsize=(13, 4))
# norm = simple_norm(npred.data, stretch="asinh", asinh_a=0.01, max_cut=350)

wcs = WCS()
height = 0.8
ax_counts = fig.add_axes([0.01, 0.12, 0.4, height], projection=wcs)
ax_npred = fig.add_axes([0.28, 0.12, 0.4, height], projection=wcs)
ax_npred_rl = fig.add_axes([0.55, 0.12, 0.4, height], projection=wcs)
ax_cbar = fig.add_axes([0.9, 0.12, 0.02, height])

ax_counts.set_title("Counts")
ax_npred.set_title("Reconstruction Jolideco")
ax_npred_rl.set_title("Reconstruction RL $N_{iter} = 100$")


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

plt.savefig(paths.figures / "example-chandra.pdf", dpi=config.DPI)
