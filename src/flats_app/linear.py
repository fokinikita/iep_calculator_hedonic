"""Linear time-dummy model, evaluated from model/linear_model.json (fitted by flats_pipeline linear.py).

    log(total price) = intercept + sum coef * (x - mean) / scale + level coefficients
                       + time effect of the region + of the okrug + of the local area (per month)

The time effects do not depend on the flat, so a flat's curve over months is its level plus the
time effects of its area.
"""

import math

import numpy as np


class LinearModel:
    def __init__(self, spec: dict):
        self.spec = spec
        self.months = spec["months"]

    def curve_log(self, row: dict) -> np.ndarray:
        """Log total price of the flat in every month. `row`: model inputs, gaps already filled
        (continuous: training mean; categorical: "missing"), plus region_type / area_okrug / local_area."""
        s = self.spec
        level = s["intercept"]
        for f in s["numeric"]:
            v = math.log(row["total_square"]) if f["name"] == s["log_area_feature"] else row[f["name"]]
            level += f["coef"] * (float(v) - f["mean"]) / f["scale"]
        other = s["other_level"]
        for c, levels in s["categorical"].items():
            v = row.get(c)
            v = "missing" if v is None else str(v)
            level += levels.get(v, levels[other])
        time = np.zeros(len(self.months))
        for kind, col in s["time_keys"].items():
            key = row.get(col)
            if kind == "region":
                key = key or "unknown"
            effects = s["time"][kind].get(key)
            if effects:
                time += np.asarray(effects)
        return level + time
