import numpy as np
import pytest
from uwb_localisation.field_boundary import CircularField, reject_outside
from uwb_localisation.empirical import EmpiricalTracker


def test_rejects_without_projection_and_preserves_valid_held_marker():
    field = CircularField(True, (0.0, 0.0), 0.5)
    method = dict(
        xy=[[0.51, 0], [0.5, 0], [0.4, 0], None],
        state=["updated"] * 4,
        reason=[""] * 4,
        boundary_rejected=[False, False, True, False],
    )
    reject_outside(method, field)
    assert method["xy"] == [None, [0.5, 0], [0.4, 0], None]
    assert method["rejected_xy"][0] == [0.51, 0]
    assert method["boundary_rejected"] == [True, False, True, False]


def test_outside_solution_does_not_initialize_or_update_state_and_recovers():
    xyz = np.array([[0.0, 0.0, 0.0], [0.866, 0.0, 0.0], [0.433, 0.75, 0.0]])
    center = np.array([0.433, 0.25])
    tr = EmpiricalTracker(xyz)
    tr.field_boundary = CircularField(True, tuple(center), 0.5)
    z = lambda xy: np.linalg.norm(xyz[:, :2] - xy, axis=1)
    row = tr.step(0, z([1.4, 0.2]))
    assert row["xy"] is None and row["boundary_rejected"]
    assert tr.x is None and tr.last_update is None
    row = tr.step(0.03, z(center))
    assert row["state"] == "updated"
    row = tr.step(0.06, z([1.4, 0.2]))
    assert row["boundary_rejected"] and row["state"] == "predicted"
    assert tr.last_update == 0.03
    assert tr.field_boundary.accepts(row["xy"])
    row = tr.step(0.09, z(center))
    assert row["state"] == "updated"


def test_prediction_cannot_escape_circle_and_history_never_smooths_invalid_state():
    xyz = np.array([[0.0, 0.0, 0.0], [0.866, 0.0, 0.0], [0.433, 0.75, 0.0]])
    tr = EmpiricalTracker(xyz)
    tr.field_boundary = CircularField(True, (0.433, 0.25), 0.5)
    tr.step(0, np.linalg.norm(xyz[:, :2] - [0.8, 0.25], axis=1))
    tr.x[2:] = [10.0, 0.0]
    row = tr.step(0.03, [np.nan] * 3)
    assert row["xy"] is None and tr.x is None
    assert tr.history[-1]["x"] is None and row["reason"] == "outside_field_rejected"


@pytest.mark.parametrize(
    "kwargs",
    [{"radius_m": 0}, {"center": [1]}, {"center": [float("nan"), 0]}, {"enabled": 1}],
)
def test_invalid_boundary(kwargs):
    with pytest.raises(ValueError):
        CircularField(**kwargs)
