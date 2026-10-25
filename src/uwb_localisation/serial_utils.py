from __future__ import annotations

from dataclasses import dataclass

from serial.tools import list_ports


@dataclass(frozen=True)
class PortInfo:
    device: str
    description: str
    hardware_id: str
    likely_kind: str


def classify_port(device: str, description: str) -> str:
    value = f"{device} {description}".lower()
    if "usbmodem" in value or "stm32" in value or "cdc" in value:
        return "USB data port"
    if "usbserial" in value or "wch" in value or "ch340" in value:
        return "TTL configuration port"
    return "Other serial port"


def available_ports() -> list[PortInfo]:
    result: list[PortInfo] = []
    for port in list_ports.comports():
        result.append(
            PortInfo(
                device=port.device,
                description=port.description or "",
                hardware_id=port.hwid or "",
                likely_kind=classify_port(port.device, port.description or ""),
            )
        )
    return sorted(result, key=lambda item: item.device)


def choose_port(preferred: str, kind: str) -> str:
    if preferred != "auto":
        return preferred

    ports = available_ports()
    if kind == "ttl":
        candidates = [
            p.device for p in ports if p.likely_kind == "TTL configuration port"
        ]
    else:
        candidates = [p.device for p in ports if p.likely_kind == "USB data port"]

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        label = "TTL configuration port" if kind == "ttl" else "USB data port"
        raise RuntimeError(f"No {label} detected. Check the USB connection and cable")
    raise RuntimeError(
        f"Multiple candidate ports. Select one with --port: {', '.join(candidates)}"
    )
