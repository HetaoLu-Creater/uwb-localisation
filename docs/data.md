# Experimental data

The data folders contain 20 recordings from physical BU03 experiments. Raw serial bytes and their serial read timestamps are unchanged. Each folder has a descriptive name and a small English metadata file.

## Recordings

| Folder | Contents | Purpose |
| --- | --- | --- |
| `data/early_0p6m` | Two repeated centre recordings and three centre recordings with different tag orientations | Check early range bias and the first offset update |
| `data/one_metre` | Nine stationary recordings and six handheld movement recordings | Fit calibration, check stationary positions and compare movement output |

The three orientation recordings in the early layout face A0, A1 and A2. Their first 5 s were used for the offset update. The later halves were used to check the result. The inherited slopes remained unchanged.

The stationary recordings in the 1 m layout cover the centre, the three side midpoints and positions near each anchor. At the centre, the tag faced each anchor in turn. Each near-anchor position was 2 cm from its anchor towards the centre. Seven recordings were used to fit calibration and two were reserved for checks. The roles are listed in `data/recordings.json`.

The six movement recordings start near an anchor and move towards the centre. Each anchor has two recordings. The first was used to select tracking settings and the second was used for comparison. Reference positions and movement labels define the evaluation conditions.

## Files in each recording

| File | Contents |
| --- | --- |
| `raw_stream.bin` | Original serial bytes received from A0 |
| `raw_chunks.csv` | The time and byte interval of each serial read |
| `session.json` | Geometry, recording duration, reference position and test role |
| `original_outputs.jsonl` | Positions produced during the original 1 m recording |

A serial read can contain part of a frame or several frames. The decoder reconstructs complete frames before processing. The device timer determines frame times. Serial read timestamps also allow reception interruptions to be examined.

`original_outputs.jsonl` contains the tracked positions saved during acquisition, using the calibration active at that time. These positions appear as Original recorded position in the replay interface. The report comparison reprocesses the raw data with the earlier version and final method, as described in [Method details](method.md).

The early orientation folders also contain `positions_before_offset_update_*.csv` and `positions_after_offset_update_*.csv`. These are saved historical processing results. The `full` files cover the complete recording. The `later_half` files contain results from processing the last 5 s with a newly initialised tracker. The evaluation script recalculates early position errors from these saved positions and refits the offset from the raw ranges.

## Repeat the numerical checks

After installing the project, run these commands from the project folder.

```sh
python scripts/evaluate.py
python scripts/fit_calibration.py
```

The first command reruns the final method on the raw 1 m recordings. It calculates stationary position RMSE, stationary variation and the percentage of movement frames with a displayed position. It also repeats the comparison with the earlier version by disabling the two refinements. Calibration, range filtering, tracking and display settings remain the same.

The second command refits the calibration curves and their range uncertainty tables. It repeats the checks using positions set aside for testing and the two reserved recordings. The original fit used a 2 cm constraint for each nearby anchor and an interval around the nominal distances to the other anchors. Reported errors use the confirmed positions 2 cm towards the centre. The fit uses the inherited slopes for regularisation to reproduce the saved calibration.

Both commands check their results against `data/expected_results.json`. Results are written to `results/` as numerical JSON files.

## Interpretation

Position RMSE uses all available displayed positions at frame times. Stationary variation uses positions after the first 0.5 s and measures their spread around the median position. A retained displayed position counts as available until it is cleared. The frame percentage uses the number of received frames as its denominator.

Movement output percentages describe the availability of displayed positions. The measurements were taken with a handheld tag.
