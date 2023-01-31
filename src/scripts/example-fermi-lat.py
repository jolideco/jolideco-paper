import config
import matplotlib.pyplot as plt
import paths

figsize = config.FigureSizeAA(aspect_ratio=1.618, width_aa="two-column")

fig, ax = plt.subplots(figsize=figsize.inch)

plt.savefig(paths.figures / "example-fermi-lat.pdf", dpi=config.DPI)