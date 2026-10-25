from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import serial


# Ai-Thinker's stock TWR scheduler uses a fixed 10 ms slot.  CAP is stored as
# ``tagnumSlots`` and the nominal superframe is CAP multiplied by this value.
BU03_TWR_SLOT_DURATION_MS = 10.0


@dataclass(frozen=True)
class ATResponse:
    command: str
    text: str

    @property
    def ok(self) -> bool:
        return any(line.strip().upper() == "OK" for line in self.text.splitlines())


def _binary_setting(value: object, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be 0 or 1") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be 0 or 1")
    parsed = int(numeric)
    if parsed not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return parsed


def _integer_setting(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def normalize_twr_device_profile(
    profile: Mapping[str, object],
) -> dict[str, int | float]:
    """Check the BU03 SETDEV fields and convert them to the required numeric types.

    One command sets the ranging superframe length, device filters and calibration.
    The same converted values are used to configure the board and check its response.
    Fields that require integers are checked before the command is sent.
    """

    required = (
        "tag_slots",
        "antenna_delay",
        "kalman_enabled",
        "kalman_q",
        "kalman_r",
        "calibration_a",
        "calibration_b_mm",
        "position_enabled",
        "position_dimension",
    )
    missing = next((name for name in required if name not in profile), None)
    if missing is not None:
        raise ValueError(f"TWR device profile is missing {missing}")

    tag_slots = _integer_setting(profile["tag_slots"], "tag_slots")
    antenna_delay = _integer_setting(profile["antenna_delay"], "antenna_delay")
    kalman_enabled = _binary_setting(profile["kalman_enabled"], "kalman_enabled")
    try:
        kalman_q = float(profile["kalman_q"])
        kalman_r = float(profile["kalman_r"])
        calibration_a = float(profile["calibration_a"])
        calibration_b_mm = float(profile["calibration_b_mm"])
    except (TypeError, ValueError) as error:
        raise ValueError("TWR device profile contains a non-numeric value") from error
    position_enabled = _binary_setting(profile["position_enabled"], "position_enabled")
    position_dimension = _integer_setting(
        profile["position_dimension"], "position_dimension"
    )

    if not 2 <= tag_slots <= 32:
        raise ValueError("tag_slots must be between 2 and 32")
    if not 0 <= antenna_delay <= 65535:
        raise ValueError("antenna_delay must be between 0 and 65535")
    finite_values = {
        "kalman_q": kalman_q,
        "kalman_r": kalman_r,
        "calibration_a": calibration_a,
        "calibration_b_mm": calibration_b_mm,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if kalman_q < 0.0 or kalman_r <= 0.0:
        raise ValueError("kalman_q must be non-negative and kalman_r must be positive")
    if calibration_a <= 0.0:
        raise ValueError("calibration_a must be positive")
    if position_dimension not in (0, 2, 3):
        raise ValueError("position_dimension must be 0, 2, or 3")
    if position_enabled == 0 and position_dimension != 0:
        raise ValueError("disabled position output requires position_dimension 0")
    if position_enabled == 1 and position_dimension not in (2, 3):
        raise ValueError("enabled position output requires dimension 2 or 3")

    return {
        "tag_slots": tag_slots,
        "antenna_delay": antenna_delay,
        "kalman_enabled": kalman_enabled,
        "kalman_q": kalman_q,
        "kalman_r": kalman_r,
        "calibration_a": calibration_a,
        "calibration_b_mm": calibration_b_mm,
        "position_enabled": position_enabled,
        "position_dimension": position_dimension,
    }


def build_setdev_command(profile: Mapping[str, object]) -> str:
    """Build one validated BU03 ``AT+SETDEV`` command."""

    normalized = normalize_twr_device_profile(profile)

    values = (
        str(normalized["tag_slots"]),
        str(normalized["antenna_delay"]),
        str(normalized["kalman_enabled"]),
        format(normalized["kalman_q"], ".9g"),
        format(normalized["kalman_r"], ".9g"),
        format(normalized["calibration_a"], ".9g"),
        format(normalized["calibration_b_mm"], ".9g"),
        str(normalized["position_enabled"]),
        str(normalized["position_dimension"]),
    )
    return "AT+SETDEV=" + ",".join(values)


def nominal_twr_frame_period_ms(tag_slots: object) -> float:
    """Return the stock-firmware nominal superframe period for a CAP value."""

    slots = _integer_setting(tag_slots, "tag_slots")
    if not 2 <= slots <= 32:
        raise ValueError("tag_slots must be between 2 and 32")
    return slots * BU03_TWR_SLOT_DURATION_MS


class BU03AT:
    def __init__(self, port: str, baud_rate: int = 115200, timeout: float = 0.2):
        self.port = port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            exclusive=True,
        )
        time.sleep(0.35)
        self.serial.reset_input_buffer()

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    def __enter__(self) -> "BU03AT":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def command(
        self,
        command: str,
        wait: float = 0.35,
        *,
        response_timeout: float | None = None,
    ) -> ATResponse:
        self.serial.reset_input_buffer()
        self.serial.write((command.strip() + "\r\n").encode("ascii"))
        self.serial.flush()

        timeout_s = (
            max(1.5, wait + 1.0)
            if response_timeout is None
            else max(0.05, float(response_timeout))
        )
        deadline = time.monotonic() + timeout_s
        chunks = bytearray()
        last_data_at: float | None = None

        while time.monotonic() < deadline:
            waiting = self.serial.in_waiting
            if waiting:
                chunks.extend(self.serial.read(waiting))
                last_data_at = time.monotonic()
            elif last_data_at is not None and time.monotonic() - last_data_at > 0.12:
                break
            time.sleep(0.02)

        return ATResponse(command, chunks.decode("utf-8", errors="replace").strip())

    def probe(
        self,
        *,
        response_timeout: float | None = None,
        configuration_only: bool = False,
        stop_if_unresponsive: bool = False,
    ) -> list[ATResponse]:
        commands = (
            ["AT", "AT+GETCFG", "AT+GETUWBMODE", "AT+GETDEV"]
            if configuration_only
            else ["AT", "AT+GETVER", "AT+GETCFG", "AT+GETUWBMODE", "AT+GETDEV"]
        )
        responses: list[ATResponse] = []
        for index, command in enumerate(commands):
            response = self.command(command, response_timeout=response_timeout)
            responses.append(response)
            if index == 0 and stop_if_unresponsive and not response.ok:
                break
        return responses

    def configure_twr(
        self,
        device_id: int,
        role: int,
        channel: int = 1,
        rate: int = 1,
        group: int | None = None,
        device_profile: Mapping[str, object] | None = None,
    ) -> list[ATResponse]:
        if not 0 <= device_id <= 10:
            raise ValueError("Device ID must be between 0 and 10")
        if role not in (0, 1):
            raise ValueError("Role must be 0 for a tag or 1 for an anchor")

        parameters = [device_id, role, channel, rate]
        if group is not None:
            parameters.append(group)

        set_cfg = "AT+SETCFG=" + ",".join(str(value) for value in parameters)
        responses = [
            self.command("AT"),
            self.command("AT+SETUWBMODE=0"),
            self.command(set_cfg),
        ]

        if device_profile is not None and all(response.ok for response in responses):
            responses.append(self.command(build_setdev_command(device_profile)))

        if not all(response.ok for response in responses):
            return responses

        responses.append(self.command("AT+SAVE", wait=0.8))
        return responses
