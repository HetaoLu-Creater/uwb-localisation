# UWB Localisation

A Python host implementation for three Ai Thinker BU03 anchors and one BU03 tag. The program reads ranges from A0, estimates a planar position and displays it in a browser. Each live run saves raw measurements and outputs for replay with the same processing method.

The project includes the final tracking method, its full precision configuration and the experimental recordings used in the UWB report. Numerical scripts check calibration, stationary position error and position display during movement.

## Install

Use Python 3.11 or later. Run the following commands from the project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Replay a recording

Replay uses the supplied recordings.

```bash
uwb replay data/one_metre/from_a2_run1
```

Open the local address printed by the program. The replay interface uses the same layout as the live Localisation Monitor. It shows positions and anchor ranges. Playback controls change the viewing time and speed. Processing uses the original measurement times.

| Position | Meaning |
| --- | --- |
| Tracked position | Position calculated from the range readings before the extra smoothing used for display. |
| Displayed position | Position shown after smoothing and display checks. It can stay visible briefly between updates. |
| Original recorded position | Tracked position saved during the original run. Replay shows it alongside the newly processed positions for comparison. |

The blue marker shows the tracked position, and the red marker shows the displayed position. Select `Original recording` to show the saved position in grey. The same explanations appear in the position guide at the bottom of the interface.

To export the processed frame results, run

```bash
uwb replay data/one_metre/from_a2_run1 --no-browser --export results/replay.jsonl
```

Use `--config` before the command to select a different processing configuration.

```bash
uwb --config configs/final.json replay data/one_metre/centre_facing_a0
```

## Run with hardware

The boards retain manufacturer firmware. See [hardware setup](docs/hardware.md) for board roles and serial connections.

```bash
uwb ports
uwb live --port /dev/cu.usbmodemXXXX --output recordings/run01
```

Replace the example port with the A0 USB data port listed on your computer. Windows ports normally use names such as `COM3`. Each run creates a new recording directory and stores the received measurements and processing results. Stop the program with Ctrl-C.

To record for 10 s from the terminal, run

```bash
uwb capture --port /dev/cu.usbmodemXXXX --seconds 10 --output recordings/run02
```

## Check the results

```bash
python scripts/evaluate.py
python scripts/fit_calibration.py
python -m pytest
```

The evaluation reprocesses the 1 m recordings with the final method and the earlier version. The earlier version changes the two refinements described in the report while keeping the other settings unchanged. The calibration script refits the per-anchor curves from the supplied stationary data.

The expected final results include stationary position RMSE from 1.55 cm to 6.79 cm. Positions were displayed for 78.32% of movement frames, compared with 74.22% for the earlier version. These results describe the supplied stationary and handheld movement recordings. The movement evaluation measures display availability and interruption handling.

## Project contents

| Location | Contents |
| --- | --- |
| `src/uwb_localisation/` | Serial input, final tracking method and live and replay interfaces. |
| `configs/final.json` | Geometry, calibration coefficients and processing settings. |
| `data/one_metre/` | Nine stationary recordings and six movement recordings. |
| `data/early_0p6m/` | Five early recordings and the saved results of the offset calibration check. |
| `scripts/` | Numerical evaluation and calibration fitting. |
| `tests/` | Protocol, algorithm, timing and replay checks. |
| `docs/` | Hardware setup, method explanation and data descriptions. |

[Method details](docs/method.md) explain the processing stages and configuration. [Data descriptions](docs/data.md) explain the reference positions, fitting split and recorded files.

## Licence and references

The original code, documentation and supplied experimental data are provided under the [MIT licence](LICENSE). [References and dependencies](THIRD_PARTY.md) identify the method references and separately installed dependencies.
