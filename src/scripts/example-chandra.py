import config
import matplotlib.pyplot as plt

figsize = config.FigureSizeAA(aspect_ratio=1.618, width_aa="two-column")

fig, ax = plt.subplots(figsize=figsize.inch)

plt.savefig("example-chandra.pdf", dpi=config.DPI)