import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from matplotlib.patches import Rectangle

random_state = np.random.RandomState(49837)
filename = get_pkg_data_filename("galactic_center/gc_msx_e.fits")


image_size = (128, 68)
x, y = 10, 40
cutout = (slice(y, y + image_size[1]), slice(x, x + image_size[0]))

figsize = config.FigureSizeAA(aspect_ratio=image_size[0] / image_size[1])

fig = plt.figure(figsize=figsize.inch)
ax = fig.add_axes([0, 0, 1, 1])

max_value = 100
data = block_reduce(fits.getdata(filename)[cutout], 2)
data = np.clip(max_value * data / data.max(), 0, np.inf)
data = random_state.poisson(data)
norm = simple_norm(data, "asinh", asinh_a=0.01, min_cut=0, max_cut=max_value)
ax.imshow(data, origin="lower", norm=norm, cmap="viridis")

image_size = (64, 32)
patch_size = (8, 8)
stride = 6

for x in range(0, image_size[0] - stride, stride):
    for y in range(0, image_size[1] - stride, stride):
        rectangle = Rectangle(
            xy=(x + 0.5, y + 0.5),
            width=patch_size[0],
            height=patch_size[1],
            facecolor="white",
            edgecolor="none",
            alpha=0.2,
        )

        ax.add_patch(rectangle)

rectangle = Rectangle(
    xy=(12.5, 12.5),
    width=patch_size[0],
    height=patch_size[1],
    facecolor="None",
    edgecolor="tab:red",
    lw=1,
)
ax.add_patch(rectangle)

ax.text(x=16.5, y=21.5, s="8 x 8 Patch", color="tab:red", ha="center")

offset = patch_size[0] // 2
# ax.set_xlim(offset, image_size[0] - offset - offset // 2)
# ax.set_ylim(offset, image_size[1] - offset)
# ax.set_aspect("equal")
plt.axis("off")
plt.savefig(paths.figures / "patches.pdf", facecolor="w", dpi=300)
