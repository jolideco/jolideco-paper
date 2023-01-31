import config
import matplotlib.pyplot as plt
import paths
from astropy.io import fits
from astropy.wcs import WCS

figsize = config.FigureSizeAA(aspect_ratio=2.5, width_aa="two-column")

wcs = WCS()
fig, ax = plt.subplots(figsize=figsize.inch, ncols=3, subplot_kw={"projection": wcs})

plt.savefig(paths.figures / "example-chandra.pdf", dpi=config.DPI)
