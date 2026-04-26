#!/usr/bin/env python3
"""
CW2 Data Collection — LiDAR Recorder
======================================
Record RP-Lidar sequences with live visualization for Q3 (Lidar SLAM).

Compatible with existing lab code (LidarDriver replay mode, ICP, occupancy grid).

Outputs:
  - scans.json          : JSON-lines scan data (same format as lab code)
  - timestamps.txt      : Timestamp per scan
  - loop_markers.json   : Manually marked loop completion events
  - sequence_info.json  : Recording metadata

Usage:
  # Live recording with RP-Lidar
  python lidar_recorder.py --name indoor_large_01 --env indoor --desc "Marshgate hallways"

  # Test with existing lab data (replay mode, no hardware needed)
  python lidar_recorder.py --name test --env indoor --mode replay \
      --input ../../Labs/Lab_08_-_Point_Cloud/Code/lab_data_01.json

Controls:
  L     : Mark loop completion (press when you return to start)
  q/ESC : Stop recording
  Ctrl+C: Stop recording (safe shutdown)
"""

import os
import sys
import time
import json
import math
import argparse
import platform
import numpy as np

# ─── RP-Lidar import ───
try:
    from rplidar import RPLidar, RPLidarException
    RPLIDAR_AVAILABLE = True
except ImportError:
    RPLIDAR_AVAILABLE = False

# ─── Visualization ───
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# ─── Constants ───
BAUD_RATE = 256000
MAX_DISPLAY_RANGE_MM = 6000


def detect_port():
    """Auto-detect RP-Lidar serial port by scanning for USB serial devices."""
    import glob

    os_name = platform.system()

    if os_name == "Darwin":
        # macOS: scan for common USB serial device patterns
        patterns = [
            "/dev/tty.usbserial*",     # FTDI / CP210x / CH340 drivers
            "/dev/tty.SLAB_USBtoUART*", # Older SiLabs driver name
            "/dev/tty.wchusbserial*",   # WCH CH340 driver
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))

        if candidates:
            port = candidates[0]
            if len(candidates) > 1:
                print(f"[INFO] Multiple USB serial ports found: {candidates}")
                print(f"[INFO] Using: {port}  (override with --port)")
            return port
        else:
            print("[WARN] No USB serial port auto-detected. Available /dev/tty.* devices:")
            for dev in sorted(glob.glob("/dev/tty.*")):
                print(f"       {dev}")
            print("[HINT] Use --port /dev/tty.YOUR_DEVICE to specify manually")
            return "/dev/tty.usbserial-0"  # fallback

    elif os_name == "Windows":
        return "COM8"
    else:
        # Linux
        candidates = glob.glob("/dev/ttyUSB*")
        return candidates[0] if candidates else "/dev/ttyUSB0"


def create_output_dir(base_dir, name):
    """Create output directory."""
    out_dir = os.path.join(base_dir, "lidar", name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


class LiveLidarRecorder:
    """Record RP-Lidar data with live polar visualization."""

    def __init__(self, out_dir, env_type, description="", port=None):
        self.out_dir = out_dir
        self.env_type = env_type
        self.description = description
        self.port = port or detect_port()

        # Data storage
        self.scan_count = 0
        self.timestamps = []
        self.loop_markers = []
        self.start_time = None

        # File handles
        self.scan_file = None
        self.ts_file = None

        # State
        self.is_running = False

    def _open_files(self):
        """Open output files for writing."""
        self.scan_file = open(
            os.path.join(self.out_dir, "scans.json"), "w"
        )
        self.ts_file = open(
            os.path.join(self.out_dir, "timestamps.txt"), "w"
        )
        self.ts_file.write("# scan_index timestamp_seconds\n")

    def _close_files(self):
        """Flush and close output files."""
        if self.scan_file:
            self.scan_file.flush()
            self.scan_file.close()
            self.scan_file = None
        if self.ts_file:
            self.ts_file.flush()
            self.ts_file.close()
            self.ts_file = None

    def _save_scan(self, scan_data, timestamp):
        """Write a single scan to file."""
        json.dump(scan_data, self.scan_file)
        self.scan_file.write("\n")
        self.ts_file.write(f"{self.scan_count} {timestamp:.6f}\n")
        self.timestamps.append(timestamp)
        self.scan_count += 1

    def mark_loop(self):
        """Mark current scan as a loop completion point."""
        marker = {
            "scan_index": self.scan_count,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.start_time if self.start_time else 0,
            "loop_number": len(self.loop_markers) + 1,
        }
        self.loop_markers.append(marker)
        print(f"\n[LOOP {marker['loop_number']}] Marked at scan {self.scan_count}, "
              f"t={marker['elapsed_seconds']:.1f}s")

    def _save_metadata(self):
        """Save recording metadata and loop markers."""
        # Loop markers
        markers_path = os.path.join(self.out_dir, "loop_markers.json")
        with open(markers_path, "w") as f:
            json.dump(self.loop_markers, f, indent=2)

        # Sequence info
        duration = (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0
        info = {
            "name": os.path.basename(self.out_dir),
            "environment": self.env_type,
            "description": self.description,
            "sensor": "RP-Lidar A2M12",
            "port": self.port,
            "baud_rate": BAUD_RATE,
            "scan_count": self.scan_count,
            "duration_seconds": round(duration, 2),
            "avg_scan_rate_hz": round(self.scan_count / duration, 2) if duration > 0 else 0,
            "loops_marked": len(self.loop_markers),
            "loop_markers": self.loop_markers,
            "timestamp_start": self.timestamps[0] if self.timestamps else None,
            "timestamp_end": self.timestamps[-1] if self.timestamps else None,
        }

        info_path = os.path.join(self.out_dir, "sequence_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        return info

    def record_live(self):
        """Record from live RP-Lidar with matplotlib polar visualization."""
        if not RPLIDAR_AVAILABLE:
            print("[ERROR] rplidar package not installed.")
            print("        Install with: pip install rplidar-roboticia")
            sys.exit(1)

        print(f"\n[INFO] Connecting to RP-Lidar on {self.port}...")
        lidar = RPLidar(self.port, baudrate=BAUD_RATE)

        # ─── Flush any stale data from previous crashed sessions ───
        # The rplidar get_health() returns a STRING (not a tuple) if there's
        # leftover data in the serial buffer, which causes a crash in start().
        try:
            lidar.clean_input()
            lidar.stop()
            time.sleep(0.5)
            lidar.clean_input()
            print("[INFO] Serial buffer flushed.")
        except Exception as e:
            print(f"[WARN] Buffer flush: {e}")

        # ─── Setup matplotlib polar plot ───
        plt.ion()
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
        fig.canvas.manager.set_window_title("CW2 LiDAR Recorder")
        ax.set_rmax(MAX_DISPLAY_RANGE_MM)
        ax.set_title("RP-Lidar Recording — Press L for loop, Q to stop", pad=20)

        # Create a persistent scatter plot that we update (avoids ax.clear() GIL crash)
        scatter = ax.scatter([], [], s=3, c="blue", alpha=0.7)

        # Status text
        status_text = fig.text(0.02, 0.02, "", fontsize=10,
                               fontfamily="monospace",
                               bbox=dict(boxstyle="round", facecolor="wheat"))

        self._open_files()
        self.start_time = time.time()
        self.is_running = True

        # Keyboard handler
        def on_key(event):
            if event.key == "l":
                self.mark_loop()
            elif event.key in ("q", "escape"):
                self.is_running = False

        fig.canvas.mpl_connect("key_press_event", on_key)

        # Handle Ctrl+C gracefully (avoid GIL crash inside matplotlib)
        import signal
        original_sigint = signal.getsignal(signal.SIGINT)

        def sigint_handler(signum, frame):
            print("\n[INFO] Ctrl+C — stopping (please wait)...")
            self.is_running = False

        signal.signal(signal.SIGINT, sigint_handler)

        print("\n" + "=" * 50)
        print("  RECORDING — RP-Lidar")
        print(f"  Environment: {self.env_type}")
        print(f"  Description: {self.description}")
        print("  Controls: L=mark loop, Q/ESC=stop")
        print("  IMPORTANT: Do exactly 2 loops, press L at each")
        print("=" * 50 + "\n")

        try:
            for scan in lidar.iter_scans():
                if not self.is_running:
                    break
                if not plt.fignum_exists(fig.number):
                    break

                timestamp = time.time()
                elapsed = timestamp - self.start_time

                # Save raw scan data
                scan_list = [[q, a, d] for q, a, d in scan]
                self._save_scan(scan_list, timestamp)

                # ─── Visualize (update scatter, no ax.clear()) ───
                valid_pts = [(math.radians(pt[1]), pt[2]) for pt in scan if pt[2] > 0]
                if valid_pts:
                    angles_rad, distances = zip(*valid_pts)
                    scatter.remove()
                    scatter = ax.scatter(angles_rad, distances, s=3, c="blue", alpha=0.7)

                # Title with loop info
                loop_str = f"Loops: {len(self.loop_markers)}/2"
                ax.set_title(f"Recording | {loop_str} | Press L=loop, Q=stop", pad=20)

                # Status
                n_pts = len(valid_pts) if valid_pts else 0
                status_text.set_text(
                    f"Scans: {self.scan_count:5d}  |  "
                    f"Time: {elapsed:6.1f}s  |  "
                    f"Points: {n_pts:4d}  |  "
                    f"Loops: {len(self.loop_markers)}"
                )

                plt.draw()
                plt.pause(0.01)

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C — stopping...")
        except RPLidarException as e:
            print(f"\n[WARN] Lidar error: {e}")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            # Cleanup
            self._close_files()
            try:
                lidar.stop()
                lidar.stop_motor()
                time.sleep(0.3)
                lidar.disconnect()
            except Exception:
                pass
            try:
                plt.close("all")
            except Exception:
                pass

        info = self._save_metadata()
        self._print_summary(info)

    def record_replay(self, input_file):
        """Record from existing JSON file (for testing without hardware)."""
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}")
            sys.exit(1)

        print(f"\n[INFO] Replay mode: {input_file}")

        # ─── Setup plot ───
        plt.ion()
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
        fig.canvas.manager.set_window_title("CW2 LiDAR Recorder (Replay)")
        ax.set_rmax(MAX_DISPLAY_RANGE_MM)

        status_text = fig.text(0.02, 0.02, "", fontsize=10,
                               fontfamily="monospace",
                               bbox=dict(boxstyle="round", facecolor="lightyellow"))

        self._open_files()
        self.start_time = time.time()
        self.is_running = True

        def on_key(event):
            if event.key == "l":
                self.mark_loop()
            elif event.key in ("q", "escape"):
                self.is_running = False

        fig.canvas.mpl_connect("key_press_event", on_key)

        try:
            with open(input_file, "r") as f:
                for line in f:
                    if not self.is_running:
                        break
                    if not plt.fignum_exists(fig.number):
                        break
                    if not line.strip():
                        continue

                    scan = json.loads(line)
                    timestamp = time.time()
                    elapsed = timestamp - self.start_time

                    # Save (re-record in our format)
                    self._save_scan(scan, timestamp)

                    # Visualize
                    angles_rad = [math.radians(pt[1]) for pt in scan if pt[2] > 0]
                    distances = [pt[2] for pt in scan if pt[2] > 0]

                    ax.clear()
                    ax.set_rmax(MAX_DISPLAY_RANGE_MM)
                    if angles_rad:
                        ax.scatter(angles_rad, distances, s=3, c="green", alpha=0.7)

                    ax.set_title(f"Replay | Loops: {len(self.loop_markers)}/2", pad=20)
                    status_text.set_text(
                        f"Scans: {self.scan_count:5d}  |  "
                        f"Time: {elapsed:6.1f}s  |  "
                        f"Loops: {len(self.loop_markers)}"
                    )

                    plt.draw()
                    plt.pause(0.05)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
        finally:
            self._close_files()
            plt.close("all")

        info = self._save_metadata()
        self._print_summary(info)

    def _print_summary(self, info):
        """Print recording summary."""
        print("\n" + "=" * 50)
        print("  LIDAR RECORDING COMPLETE")
        print("=" * 50)
        print(f"  Scans:     {info['scan_count']}")
        print(f"  Duration:  {info['duration_seconds']}s")
        print(f"  Scan rate: {info['avg_scan_rate_hz']} Hz")
        print(f"  Loops:     {info['loops_marked']}")
        print(f"  Output:    {self.out_dir}/")
        if info["loops_marked"] < 2:
            print(f"  [WARN] CW2 requires exactly 2 loops!")
        else:
            print(f"  [OK] Loop requirement met")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="CW2 LiDAR Recorder — Record RP-Lidar sequences"
    )
    parser.add_argument(
        "--name", required=True,
        help="Sequence name (e.g., 'indoor_large_01', 'outdoor_01')"
    )
    parser.add_argument(
        "--env", required=True, choices=["indoor", "outdoor"],
        help="Environment type"
    )
    parser.add_argument(
        "--desc", default="",
        help="Description of the environment being scanned"
    )
    parser.add_argument(
        "--mode", default="live", choices=["live", "replay"],
        help="Recording mode (default: live)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Input file for replay mode"
    )
    parser.add_argument(
        "--port", default=None,
        help="Serial port override (default: auto-detect)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Base data directory (default: data/)"
    )

    args = parser.parse_args()

    if args.mode == "replay" and args.input is None:
        print("[ERROR] --input required for replay mode")
        sys.exit(1)

    out_dir = create_output_dir(args.data_dir, args.name)

    # Check for existing data
    if os.path.exists(os.path.join(out_dir, "scans.json")):
        print(f"[WARN] Data already exists in {out_dir}")
        resp = input("Overwrite? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    recorder = LiveLidarRecorder(
        out_dir=out_dir,
        env_type=args.env,
        description=args.desc,
        port=args.port,
    )

    if args.mode == "live":
        recorder.record_live()
    else:
        recorder.record_replay(args.input)


if __name__ == "__main__":
    main()
