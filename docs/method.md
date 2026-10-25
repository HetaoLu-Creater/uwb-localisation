# Localisation method

The final method estimates a planar tag position from three ranges and a fixed tag height. A range is a measured distance from an anchor to the tag. Each decoded frame contains the ranges for one measurement update. Processing uses current and past measurements, so the same method runs live and during replay.

## Processing stages

1. **Range filtering.** Each link has a filter for range and its rate of change. Stable readings from at least two links support the stationary model. Joint changes support the moving model. Isolated jumps are withheld until the changes persist.
2. **Range calibration.** Each anchor has a continuous calibration function with two linear segments. The coefficients convert filtered ranges into calibrated ranges. Negative calibrated values remain visible to the residual checks.
3. **Position estimation.** A weighted nonlinear least squares solver finds a planar position consistent with the three calibrated ranges. Weights account for random uncertainty, persistent bias and correlation between nearby measurements. A Huber loss reduces the influence of large residuals.
4. **Motion tracking.** A Kalman filter tracks position and velocity using the interval between device timestamps. It rejects inconsistent updates and positions outside the circular field. Predictions expire after 0.3 s.
5. **Smoothing and display.** An exponential filter smooths the tracked position with a time constant of 0.4 s. Display updates run at a nominal 5 Hz. A new displayed position requires an accepted measurement, acceptable residuals and completion of the initial settling period. A held position is cleared when its source measurement is older than 0.4 s. A tracker reset can clear it earlier.

The tracked position is the position before display smoothing. The displayed position is the position visible after smoothing and display checks. A visible held position counts towards the displayed frame percentage. The percentage uses received frames as its denominator.

## Configuration

`configs/final.json` contains the full precision coefficients and processing settings used for the final report results. The report rounds its coefficient table for presentation. The configuration retains the original numeric precision for replay.

The nominal field has a radius of 0.5 m. A0 is the origin, A1 lies on the positive X axis and A2 lies on the positive Y side. Coordinates and ranges use metres internally. The interface displays centimetres.

Calibration follows `z = a*r + b + c*max(0, k-r)`. In the configuration, `calibration_b_mm` stores the offset in millimetres, and `calibration_hinge_knot_m` stores the breakpoint in metres. Runtime processing converts the offset to metres. Both slopes must be positive.

The uncertainty tables contain separate random and bias terms for each anchor. The method interpolates their variances as the calibrated range changes. Near an anchor, it applies an additional uncertainty adjustment during both stationary periods and movement.

## Comparison with the earlier version

The earlier version uses the same calibration, range filtering, tracking and display settings. It differs in two places. The earlier version applies near-anchor uncertainty adjustment during stationary periods and checks the circular boundary at display. The evaluation script makes these two changes in the same implementation.

## Code map

| File | Purpose |
| --- | --- |
| `protocol.py`, `packet_clock.py` | Decode serial frames and assign measurement times. |
| `adaptive_ranges.py` | Filter ranges and identify stable or changing input. |
| `contact_calibration.py` | Apply the per-anchor calibration and uncertainty tables. |
| `empirical.py`, `localisation.py` | Solve positions and track position and velocity. |
| `reliable_tracker.py`, `field_boundary.py` | Combine the stages and apply reliability and field checks. |
| `output_filter.py` | Smooth positions and control display updates. |
| `runtime_tracker.py`, `pipeline.py` | Provide the shared live and replay processing interface. |
| `cli.py`, `at.py`, `serial_utils.py` | Run the program and communicate with the boards. |
| `web/` | Display live and replayed positions in a browser. |

The supplied recordings cover stationary positions and handheld movement. Movement results measure display availability and interruption handling.
