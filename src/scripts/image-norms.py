import config
import matplotlib.pyplot as plt
import numpy as np
import paths
from jolideco.utils.norms import NORMS_REGISTRY

figsize = config.FigureSizeAA(aspect_ratio=1.618)

init_kwargs = {
    "fixed-max": {"max_value": 10},
    "asinh": {"alpha": 0.1, "beta": 10},
    "atan": {},
}

fig = plt.figure(figsize=figsize.inch)
ax = fig.add_axes([0.15, 0.2, 0.80, 0.77])

x = np.linspace(0, 15, 100)

for name, norm_cls in NORMS_REGISTRY.items():
    print(name)
    if name not in init_kwargs:
        continue

    norm = norm_cls(**init_kwargs[name])
    ax.plot(x, norm(x), label=name)

ax.set_xlim(0, 15)
ax.set_ylim(0, 1.2)
ax.set_xlabel("Pixel value")
ax.set_ylabel("Normalised pixel value")
plt.legend()
plt.savefig(paths.figures / "image-norms.pdf", facecolor="w", dpi=300)
