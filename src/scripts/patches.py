import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paths
import config

figsize = config.FigureSizeAA(aspect_ratio=2)

fig = plt.figure(figsize=figsize.inch)
ax = fig.add_axes([0, 0, 1, 1])

image_size = (64, 32)
patch_size = (8, 8)

for x in range(0, 2 * image_size[0], 6):
    for y in range(0, 2 * image_size[1], 6):
        rectangle = Rectangle(
            xy=(x, y),
            width=patch_size[0],
            height=patch_size[1],
            facecolor=config.COLORS[1],
            edgecolor="k",
            alpha=0.2,
        )

        ax.add_patch(rectangle)

rectangle = Rectangle(
    xy=(12, 12),
    width=patch_size[0],
    height=patch_size[1],
    facecolor="None",
    edgecolor=config.COLORS[-1],
    lw=1,
)
ax.add_patch(rectangle)

ax.text(x=16, y=20.5, s="8 x 8 Patch", color=config.COLORS[-1], ha="center")

offset = patch_size[0] // 2
ax.set_xlim(offset, image_size[0] - offset - offset // 2)
ax.set_ylim(offset, image_size[1] - offset)
ax.set_aspect("equal")
plt.axis("off")
plt.savefig(paths.figures / "patches.pdf", facecolor="w", dpi=300)
