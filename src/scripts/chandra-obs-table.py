import logging
from io import StringIO

import numpy as np
import paths
import yaml
from astropy import units as u
from astropy.table import Table
from dateutil import parser

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


colnames = ["Obs ID", "Obs Date", "Exposure"]
units = ["", "", "ks"]
dtype = [int, "S10", "f8"]
obs_table = Table(names=colnames, dtype=dtype, units=units)


path = paths.jolideco_repo_chandra_example / "data"

with (path.parent / "config/config-e0102.yaml").open("r") as fh:
    config = yaml.safe_load(fh)

obs_ids = config["chandra-data"]["obs_ids"]
OBS_ID_REF = config["chandra-data"]["obs_id_ref"]

for obs_id in obs_ids:
    filename = path / f"{obs_id}" / "oif.fits"
    table = Table.read(filename)
    exposure = table.meta["EXPOSURE"] * u.s
    date = parser.parse(table.meta["DATE-OBS"])
    row = [obs_id, date.strftime("%m/%d/%Y"), round(exposure.to_value("ks"), 1)]
    obs_table.add_row(row)


# Add reference observation
obs_table.add_index("Obs ID")
exposure_ref = obs_table.loc[OBS_ID_REF]["Exposure"]
obs_table["Rel. Exposure"] = np.round(obs_table["Exposure"] / exposure_ref, 1)

exposure_total = np.sum(obs_table["Exposure"])
obs_table["Frac. Exposure"] = np.round(100 * obs_table["Exposure"] / exposure_total, 1)
obs_table["Frac. Exposure"].unit = "%"

content_io = StringIO()
obs_table.write(content_io, format="latex", overwrite=True)

content = content_io.getvalue()
lines = content.split("\n")
lines = lines[2:-3]
lines.insert(1, "\hline")
lines.insert(3, "\hline")
lines.append("\hline")
content = "\n".join(lines)

filename = paths.output / "chandra-obs-table.tex"

with filename.open("w") as f:
    f.write(content)
