import config
import matplotlib.pyplot as plt
import numpy as np
import paths

figsize = config.FigureSizeAA(aspect_ratio=1.618)

fig = plt.figure(figsize=figsize.inch)
ax = fig.add_axes([0.1, 0.1, 0.88, 0.88])

x = np.linspace(0, 2, 100)
ax.plot(x, x, label="Linear")
ax.plot(x, x**0.5, label="Square Root")
ax.plot(x, np.arcsinh(x), label="Arcsinh")
ax.plot(x, np.clip(x, 0, 1), label="Clip")

plt.legend()
plt.savefig(paths.figures / "image-norms.pdf", facecolor="w", dpi=300)
