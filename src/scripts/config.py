"""General configuration for mpl plotting scripts"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units import imperial
from cycler import cycler

BASE_PATH = Path(__file__).parent.parent

FONTSIZE = 10
DPI = 300

FIGURE_WIDTH_AA = {
    "single-column": 90 * u.mm,
    "two-column": 180 * u.mm,
    "intermediate": 120 * u.mm,
}

CMAP_NAME = "viridis"
CMAP = plt.get_cmap(CMAP_NAME)

COLORS = {
    "white": "#ffffff",
    "red": [216 / 255, 27 / 255, 96 / 255, 1.0],
}

for idx, color in enumerate(CMAP([0, 0.25, 0.5, 0.75, 1])):
    COLORS[f"{CMAP_NAME}-{idx}"] = color

# COLORS_CYCLE = [f"{CMAP_NAME}-{idx}" for idx in range(5)]

# PROP_CYCLER = cycler(color=mcolors.TABLEAU_COLORS) + cycler(
#     linestyle=["-", "--", ":", "-.", "."]
# )


class FigureSizeAA:
    """Figure size A&A"""

    def __init__(self, aspect_ratio=1, width_aa="single-column"):
        self.width = FIGURE_WIDTH_AA[width_aa]
        self.height = self.width / aspect_ratio
        self.aspect_ratio = aspect_ratio

    @property
    def inch(self):
        """Figure size in inch"""
        return self.width.to_value(imperial.inch), self.height.to_value(imperial.inch)

    @property
    def mm(self):
        """Figure size in mm"""
        return self.width.value, self.height.value
