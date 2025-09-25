#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

import socket
import multiprocessing
import threading
import traceback
from queue import Empty, Queue
import numpy as np
import cv2

# Constants
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


def next_global_identifier(width: int = 6) -> str:
    """Generate a zero-padded incremental identifier across all tests."""
    counter_file = OUTPUT_DIR / ".global_counter.txt"
    try:
        current = int(counter_file.read_text().strip())
    except Exception:
        current = 0
    next_val = current + 1
    try:
        counter_file.write_text(str(next_val))
    except Exception:
        pass
    return f"{next_val:0{width}d}"


def build_session_dir(test_name: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    global_id = next_global_identifier()
    session_name = f"{global_id}_{test_name}_{timestamp}"
    session_dir = OUTPUT_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def receiver_process(bot_name: str, port: int, output_dir: Path):
    """Receiver process function - simplified to one output per bot."""
    def process_frame_worker(frame_queue, output_path):
        action_data = []
        frames = []

        while True:
            try:
                data = frame_queue.get(timeout=5)
                if data is None:
                    break

                img, pos = data
                img_count = pos.get("frame_count", 0)

                pos["x"] = round(pos["x"], 3)
                pos["y"] = round(pos["y"], 3)
                pos["z"] = round(pos["z"], 3)
                pos["yaw"] = round(pos["yaw"], 3)
                pos["pitch"] = round(pos["pitch"], 3)
                action_data.append(pos)

                if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                    print(f"[{bot_name}] Error: Bad image at frame {img_count}")
                    continue

                frames.append(img)

            except Empty:
                continue
            except Exception as e:
                print(f"[{bot_name}] Error processing frame: {e}")
                traceback.print_exc()
                continue

        # FPS calculation
        if len(action_data) > 1:
            real_fps = len(action_data) / (
                (action_data[-1]["renderTime"] - action_data[0]["renderTime"]) / 1000
            )
            video_fps = min(real_fps, 20)
        else:
            video_fps = 20

        out = cv2.VideoWriter(
            f"{output_path}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (640, 360),
        )
        for frame in frames:
            out.write(frame)
        out.release()

        with open(output_path + ".json", "w") as f:
            json.dump(action_data, f)

        print(f"[{bot_name}] Saved {output_path}.mp4")

    def recvall(sock, count):
        buf = b""
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def recvint(sock):
        data = recvall(sock, 4)
        if data is None:
            return 0
        return int.from_bytes(data, byteorder="little")

    HOST = ""
    PORT = port

    os.makedirs(output_dir, exist_ok=True)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(10)
    print(f"[{bot_name}] Listening on port {PORT}")

    while True:
        frame_queue = Queue()
        output_path = str(output_dir / bot_name)

        processor = threading.Thread(
            target=process_frame_worker, args=(frame_queue, output_path), daemon=True
        )
        processor.start()
        conn, addr = s.accept()
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        print(f"[{bot_name}] Connected")

        img_count = 0
        try:
            while True:
                pos_length = recvint(conn)
                if pos_length == 0:
                    break

                pos_data = recvall(conn, pos_length)
                if pos_data is None:
                    break
                pos = json.loads(pos_data.decode("utf-8"))
                pos["frame_count"] = img_count

                length = recvint(conn)
                if length == 0:
                    break
                stringData = recvall(conn, int(length))
                if stringData is None:
                    break

                img_count += 1
                img = cv2.imdecode(np.frombuffer(stringData, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                frame_queue.put((img, pos))

        except Exception as e:
            print(f"[{bot_name}] Error: {e}")
            traceback.print_exc()
        finally:
            frame_queue.put(None)
            processor.join()
            conn.close()


def write_session_meta(session_dir: Path, test_name: str, bots: List[str], receiver_base: int) -> None:
    meta = {
        "test": test_name,
        "created": datetime.datetime.now().isoformat(),
        "bots": bots,
        "receiver_port_base": receiver_base,
    }
    with open(session_dir / "session.json", "w") as f:
        json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervisor: spawn receivers per bot and organize outputs")
    p.add_argument("--bots", required=True, help="Comma-separated bot names, e.g., Alpha,Bravo")
    p.add_argument("--test", required=True, help="Test name for this run")
    p.add_argument("--receiver_port_base", type=int, default=8090)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bots = [b.strip() for b in args.bots.split(",") if b.strip()]
    if not bots:
        print("[supervisor] No bots specified")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_dir = build_session_dir(args.test)
    write_session_meta(session_dir, args.test, bots, args.receiver_port_base)

    receiver_processes: Dict[str, multiprocessing.Process] = {}
    for i, bot in enumerate(bots):
        bot_dir = session_dir / bot
        bot_dir.mkdir(parents=True, exist_ok=True)
        port = args.receiver_port_base + i
        proc = multiprocessing.Process(
            target=receiver_process, args=(bot, port, bot_dir), name=f"receiver-{bot}"
        )
        proc.start()
        receiver_processes[bot] = proc

    terminating = False

    def handle_sigterm(signum, frame):
        nonlocal terminating
        if not terminating:
            print("[supervisor] Caught termination signal; shutting down...")
            terminating = True

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        while not terminating:
            time.sleep(2)
    finally:
        print("[supervisor] Stopping receiver processes...")
        for proc in receiver_processes.values():
            try:
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass

    print(f"[supervisor] Session artifacts at {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
