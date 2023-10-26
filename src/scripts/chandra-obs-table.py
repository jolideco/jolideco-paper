import logging
from io import StringIO

import numpy as np
import paths
from astropy import units as u
from astropy.table import Table
from dateutil import parser

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
OBS_ID_REF = 8365

colnames = ["Obs ID", "Exposure", "Obs Date"]
obs_table = Table(names=colnames, dtype=[int, "f8", "S10"], units=["", "ks", ""])

path = paths.jolideco_repo_chandra_example / "data"
filenames = sorted(path.glob("*/oif.fits"))

for filename in filenames:
    obs_id = filename.parent.name
    table = Table.read(filename)
    exposure = table.meta["EXPOSURE"] * u.s
    date = parser.parse(table.meta["DATE-OBS"])
    row = [obs_id, round(exposure.to_value("ks"), 1), date.strftime("%m/%d/%Y")]
    obs_table.add_row(row)


# Add reference observation
obs_table.add_index("Obs ID")
exposure_ref = obs_table.loc[OBS_ID_REF]["Exposure"]
obs_table["Rel. Exposure"] = np.round(obs_table["Exposure"] / exposure_ref, 1)

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
