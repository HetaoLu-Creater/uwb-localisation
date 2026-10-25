# Hardware setup

The system uses four Ai Thinker BU03 development boards. T0 is the moving tag, and A0, A1 and A2 are fixed anchors. The boards run manufacturer firmware for ranging and communication. The Python program performs the host processing.

## Configure the boards

Use the TTL configuration port to set each board role. Connect one board at a time and select its port from the list.

```bash
uwb ports
uwb probe --port /dev/cu.usbserialXXXX
uwb configure T0 --port /dev/cu.usbserialXXXX
```

Replace `T0` with `A0`, `A1` or `A2` for each anchor. The configuration command sets the role and radio configuration, saves the settings and reads them back. Use the manufacturer manual linked in [References and dependencies](../THIRD_PARTY.md) when working with another firmware revision.

The supplied processing configuration uses the raw range readings from each frame. Host calibration is applied once, after host range filtering. Board calibration should therefore retain unit scale and zero offset.

The board firmware determines the sampling schedule. Measurement intervals are recovered from device timestamps, and replay retains the timing of each supplied recording.

## Connect A0 for measurements

Power all four boards. Connect the USB data port of A0 to the computer running the host program. The TTL configuration port and USB measurement port have different purposes. The program reads measurements from the selected A0 port.

```bash
uwb ports
uwb live --port /dev/cu.usbmodemXXXX --output recordings/run01
```

Open the local address printed by the program. Live position display and recording run together. Raw measurements are saved throughout live operation.

## Place the anchors

The supplied configuration describes the 1 m circular layout used in the report. The three anchors form an equilateral triangle with side length 0.866 m. A0 is the coordinate origin, A1 lies along positive X and A2 lies on the positive Y side.

Keep the antenna heights consistent with `tag_height_m` and the anchor heights in the configuration. The tag and anchors had the same nominal height in the supplied recordings. For another layout, update the anchor coordinates and field boundary, then collect reference measurements to check the calibration.

Use `configs/final.json` for the supplied 1 m recordings. The early 0.6 m recordings have their own geometry and saved historical results, as described in [Data descriptions](data.md).
