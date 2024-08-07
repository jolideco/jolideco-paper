import logging
from io import StringIO

import paths
import yaml
from astropy.table import Table

log = logging.getLogger(__name__)


METRICS = ["MSE", "NMI", "SSI", "NRMSE"]
SCENARIOS = {
    "spiral1": "D1",
    "spiral2": "D2",
    "spiral3": "D3",
    "spiral4": "D4",
    "spiral5": "D5",
}

METHOD = "jolideco-patch-prior-gleam-v0.1"

bkg_level_titles = {
    "bg1": "Bkg 1",
    "bg2": "Bkg 2",
    "bg3": "Bkg 3",
}


def get_metrics(path):
    with open(path) as f:
        metrics = yaml.safe_load(f)

    return metrics


def get_parameters(filename):
    parts = filename.parts
    return {
        "Method": parts[-2],
        "Bkg. Level": parts[-4],
        "Scenario": parts[-5],
        "Instrument": parts[-3],
    }


colnames = [
    "Method",
    "Bkg. Level",
    "Scenario",
    "Instrument",
]
names = colnames + METRICS

dtypes = ["S20"] * len(colnames) + ["f8"] * len(METRICS)

table = Table(names=names, dtype=dtypes)

filenames = paths.jolideco_repo_comparison.glob("results/*/bg*/*/*/metrics.yaml")


for filename in filenames:
    data = get_metrics(filename)
    data.update(get_parameters(filename))
    table.add_row(data)


selection = (table["Instrument"] == "chandra") & (table["Method"] == METHOD)
table = table[selection]


table_metrics = Table(
    names=["Scenario"] + list(bkg_level_titles.values()),
    dtype=["S2"] + ["S10"] * len(bkg_level_titles),
)

for scenario in SCENARIOS:
    data = table[table["Scenario"] == scenario]

    values = {}
    for bkg_level, title in bkg_level_titles.items():
        selection = data["Bkg. Level"] == bkg_level
        v = [float(data[selection][name]) for name in ["SSI", "NRMSE"]]
        values[title] = f"{v[0]:.2f} / {v[1]:.2f}"

    values["Scenario"] = SCENARIOS[scenario]
    table_metrics.add_row(values)


content_io = StringIO()
table_metrics.write(content_io, format="latex", overwrite=True)

content = content_io.getvalue()
lines = content.split("\n")
lines = lines[2:-3]
lines.insert(1, "\hline")
lines.append("\hline")
content = "\n".join(lines)

filename = paths.output / "metrics-table-bkg.tex"

with filename.open("w") as f:
    log.info(f"Writing {filename}")
    f.write(content)
