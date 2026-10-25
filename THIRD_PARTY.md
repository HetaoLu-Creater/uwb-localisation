# References and dependencies

## Method references

The range solver and motion tracker were developed with reference to the
[DipFlip positioning implementation](https://github.com/DipFlip/ultra-wideband-positioning/blob/ea80cb7370f207a73a8322963321158b39f8ca7a/position_solver.py)
at commit `ea80cb7370f207a73a8322963321158b39f8ca7a`. This project adapts the
range residual and position and velocity tracking approach to three anchors
and a fixed tag height. The local implementation uses separate range
uncertainties, actual measurement intervals and the configured circular field.

The algebraic starting point for the solver uses differences between squared
ranges. The [Core Electronics UWB guide](https://core-electronics.com.au/guides/diy-2d-and-3d-spatial-tracking-with-ultra-wideband-arduino-and-pico-guide/)
provided a reference for this calculation.

The BU03 boards run manufacturer firmware. Configuration commands follow the
[Ai Thinker BU03/BU04 AT command manual](https://aithinker-static.oss-cn-shenzhen.aliyuncs.com/docs/Public%20Document%20Center/UWB/uwb_1/Support/BU03_BU04_AT_command_en_v1.0.6.pdf).
The links above lead to the original sources and manufacturer documentation.

## Python dependencies

Dependencies are installed separately through `pip` and retain their own
licences. The main project licences are listed below. Installed distributions
may include additional notices for bundled components.

| Dependency | Main project licence | Source |
| --- | --- | --- |
| NumPy | BSD 3-Clause | [Licence](https://numpy.org/doc/stable/license.html) |
| SciPy | BSD 3-Clause | [Licence](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| pySerial | BSD 3-Clause | [Licence](https://github.com/pyserial/pyserial/blob/master/LICENSE.txt) |
| pytest | MIT | [Licence](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
