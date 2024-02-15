import logging
from io import StringIO

import paths
import yaml
from astropy.table import Table

log = logging.getLogger(__name__)


METRICS = ["MSE", "NMI", "SSI", "NRMSE"]
SCENARIOS = {"point1": "A1", "aster3": "B3", "disk3": "C3", "spiral4": "D4"}


method_titles = {
    # "gt": "Ground Truth",
    "jolideco-uniform-prior=n=10": "Jolideco\n(Uni, n=10)",
    "jolideco-uniform-prior=n=1000": "Jolideco\n(Unif., n=1000)",
    "pylira": "Pylira",
    "jolideco-patch-prior-zoran-weiss": "Jolideco\n(Zoran-Weiss)",
    "jolideco-patch-prior-gleam-v0.1": "Jolideco\n(GLEAM v0.1)",
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


selection = (table["Instrument"] == "chandra") & (table["Bkg. Level"] == "bg1")
table = table[selection]


table_metrics = Table(
    names=["Scenario"] + list(method_titles.values()),
    dtype=["S2"] + ["S10"] * len(method_titles),
)

for scenario in SCENARIOS:
    data = table[table["Scenario"] == scenario]

    values = {}
    for method, title in method_titles.items():
        selection = data["Method"] == method
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

filename = paths.output / "metrics-table.tex"

with filename.open("w") as f:
    log.info(f"Writing {filename}")
    f.write(content)
