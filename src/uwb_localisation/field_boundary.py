"""Reject position estimates outside a circular field."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CircularField:
    enabled: bool = False
    center: tuple = (0.0, 0.0)
    radius_m: float = 0.5

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("field_boundary.enabled must be boolean")
        if np.asarray(self.center).shape != (2,) or not np.isfinite(self.center).all():
            raise ValueError("field_boundary.center needs two finite coordinates")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("field_boundary.radius_m must be positive and finite")

    def accepts(self, xy):
        if xy is None:
            return False
        point = np.asarray(xy, float)
        if point.shape != (2,) or not np.isfinite(point).all():
            return False
        return (
            not self.enabled
            or np.linalg.norm(point - self.center) <= self.radius_m + 1e-9
        )


def reject_outside(method, field):
    """Reject positions outside the field in an evaluation trace.

    Recompute availability metrics after applying this check. Display rules may
    retain an earlier accepted position while a new candidate is rejected.
    """
    n = len(method["xy"])
    flags = [bool(v) for v in method.get("boundary_rejected", [False] * n)]
    rejected = list(method.get("rejected_xy", [None] * n))
    method["xy"] = list(method["xy"])
    method["state"] = list(method.get("state", ["updated"] * n))
    method["reason"] = list(method.get("reason", [""] * n))
    for k, xy in enumerate(method["xy"]):
        if xy is not None and not field.accepts(xy):
            rejected[k] = np.asarray(xy).tolist()
            method["xy"][k] = None
            method["state"][k] = "lost"
            method["reason"][k] = "outside_field_rejected"
            flags[k] = True
            for key in ("covariance", "bias_covariance"):
                if key in method:
                    method[key] = list(method[key])
                    method[key][k] = None
    method["boundary_rejected"] = flags
    method["rejected_xy"] = rejected
    return method
