import config
import matplotlib.pyplot as plt
import paths
from jolideco.priors import GaussianMixtureModel

figsize = config.FigureSizeAA(aspect_ratio=1.618)

fig = plt.figure(figsize=figsize.inch)

gmm = GaussianMixtureModel.from_registry("nrao-jets-v0.1")


plt.legend()
plt.savefig(paths.figures / "gmm-eigen-images.pdf", facecolor="w", dpi=300)
