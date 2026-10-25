"""Commands for board configuration, live processing and recording replay."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import serial

from .at import BU03AT
from .pipeline import ProcessingPipeline, json_safe, replay_recording
from .serial_utils import available_ports


def load_config(path):
    config = json.loads(Path(path).read_text())
    ProcessingPipeline(config)
    return config


def serve(config, state, trace, lock, port):
    assets = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/config":
                payload = dict(
                    anchors=config["anchors"],
                    field=config.get("field_boundary", {}),
                    maximum_age_s=config["reliable_output"]["maximum_source_age_s"],
                    prediction_age_s=config["empirical_filter"][
                        "maximum_prediction_age_s"
                    ],
                    replay=trace is not None,
                )
            elif self.path == "/api/state":
                with lock:
                    payload = state()
            elif self.path == "/api/trace" and trace is not None:
                payload = trace
            elif self.path in ("/", "/app.js", "/app.css"):
                name = "index.html" if self.path == "/" else self.path[1:]
                content = (assets / name).read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "app.css": "text/css",
                    }[name],
                )
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404)
                return
            content = json.dumps(json_safe(payload), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://127.0.0.1:{port}/ in a browser. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def capture(args, config, live=False):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    pipeline = ProcessingPipeline(config)
    lock, stop = threading.Lock(), threading.Event()
    failure = []
    started = time.monotonic()
    metadata = dict(
        config=config,
        duration_s=0.0,
        started_at=datetime.now(timezone.utc).isoformat(),
        nominal_packet_interval_s=pipeline.clock.nominal_interval_s,
        status="running",
    )
    (directory / "session.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def worker():
        try:
            with (
                serial.Serial(args.port, args.baud, timeout=0.05) as connection,
                (directory / "raw_stream.bin").open("wb") as raw,
                (directory / "raw_chunks.csv").open("w", newline="") as chunks,
                (directory / "outputs.jsonl").open("w") as outputs,
            ):
                writer = csv.writer(chunks)
                writer.writerow(
                    [
                        "chunk_index",
                        "timestamp_epoch_s",
                        "elapsed_s",
                        "timestamp_monotonic_s",
                        "byte_offset",
                        "length_bytes",
                    ]
                )
                offset, index = 0, 0
                while not stop.is_set() and (
                    args.seconds <= 0 or time.monotonic() - started < args.seconds
                ):
                    data = connection.read(max(1, connection.in_waiting))
                    now, epoch = time.monotonic(), time.time()
                    if not data:
                        continue
                    raw.write(data)
                    writer.writerow(
                        [index, epoch, now - started, now, offset, len(data)]
                    )
                    index += 1
                    offset += len(data)
                    with lock:
                        rows = pipeline.feed(data, now, epoch)
                    for row in rows:
                        outputs.write(json.dumps(row, allow_nan=False) + "\n")
                metadata["status"] = "completed"
        except Exception as error:
            failure.append(str(error))
            metadata["status"] = "failed"
            metadata["error"] = str(error)
        finally:
            stop.set()
            metadata["duration_s"] = time.monotonic() - started
            metadata["frames"] = pipeline.frame_count
            (directory / "session.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        if live:

            def snapshot():
                result = pipeline.snapshot(time.monotonic())
                result["running"] = not stop.is_set()
                result["error"] = failure[0] if failure else None
                return result

            serve(config, snapshot, None, lock, args.http_port)
        else:
            print(f"Recording to {directory}. Press Ctrl-C to stop.")
            while thread.is_alive():
                thread.join(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        thread.join()
    if failure:
        raise RuntimeError(failure[0])
    print(f"Saved {pipeline.frame_count} decoded frames to {directory}.")


def main():
    parser = argparse.ArgumentParser(description="BU03 ranging, tracking and replay")
    parser.add_argument(
        "--config", default="configs/final.json", help="Processing configuration"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ports", help="List serial ports")
    for command in ("probe", "configure"):
        p = sub.add_parser(command)
        p.add_argument("--port", required=True, help="TTL configuration port")
        p.add_argument("--baud", type=int, default=115200)
        if command == "configure":
            p.add_argument("role", choices=["T0", "A0", "A1", "A2"])
    for command in ("capture", "live"):
        p = sub.add_parser(command)
        p.add_argument("--port", required=True, help="A0 USB data port")
        p.add_argument("--baud", type=int, default=115200)
        p.add_argument(
            "--seconds",
            type=float,
            default=0.0,
            help="Stop after this duration; zero runs until interrupted",
        )
        p.add_argument("--output", required=True, help="New recording directory")
        if command == "live":
            p.add_argument("--http-port", type=int, default=8765)
    p = sub.add_parser("replay", help="Process a saved raw recording")
    p.add_argument("recording")
    p.add_argument("--export", help="Write processed frame results as JSON Lines")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Export without serving the replay interface",
    )
    p.add_argument("--http-port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "ports":
            for port in available_ports():
                print(f"{port.device}\t{port.likely_kind}\t{port.description}")
            return
        if args.command == "probe":
            with BU03AT(args.port, args.baud) as board:
                for response in board.probe():
                    print(f"{response.command}\n{response.text}")
            return
        config = load_config(args.config)
        if args.command == "configure":
            profile = config.get("twr_device_profile")
            with BU03AT(args.port, args.baud) as board:
                responses = board.configure_twr(
                    device_id=int(args.role[1]),
                    role=int(args.role[0] == "A"),
                    channel=config.get("radio", {}).get("channel", 1),
                    rate=config.get("radio", {}).get("rate", 1),
                    group=config.get("radio", {}).get("group"),
                    device_profile=profile,
                )
                for response in responses:
                    print(f"{response.command}\n{response.text}")
                if not all(response.ok for response in responses):
                    raise RuntimeError("Board configuration was not acknowledged")
                for response in board.probe(configuration_only=True):
                    print(f"{response.command}\n{response.text}")
            return
        if args.command in ("capture", "live"):
            capture(args, config, live=args.command == "live")
            return
        rows = replay_recording(args.recording, config)
        if args.export:
            path = Path(args.export)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
            )
            print(f"Exported {len(rows)} frames to {path}.")
        if not args.no_browser:
            original = Path(args.recording) / "original_outputs.jsonl"
            if not original.exists():
                original = Path(args.recording) / "outputs.jsonl"
            original_rows = (
                [
                    json.loads(line)
                    for line in original.read_text().splitlines()
                    if line.strip()
                ]
                if original.exists()
                else []
            )
            trace = dict(
                rows=rows,
                original_rows=original_rows,
                duration_s=max((r["time_s"] for r in rows), default=0.0)
                + config["reliable_output"]["maximum_source_age_s"],
            )
            serve(config, lambda: {}, trace, threading.Lock(), args.http_port)
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, RuntimeError, serial.SerialException) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
