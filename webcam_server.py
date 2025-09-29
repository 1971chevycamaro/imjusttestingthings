"""Shared-memory webcam frame broadcaster."""

from __future__ import annotations

import argparse
import signal
import sys
from multiprocessing import shared_memory
from typing import Optional

import cv2
import numpy as np

SHAPE_DTYPE = np.int32
FRAME_DTYPE = np.uint8


def _cleanup_shared_memory(*segments: Optional[shared_memory.SharedMemory]) -> None:
    """Close and unlink the provided shared memory segments if they exist."""

    for segment in segments:
        if segment is None:
            continue
        try:
            segment.close()
        finally:
            try:
                segment.unlink()
            except FileNotFoundError:
                # The segment may already be unlinked by another process.
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=0,
        help="Camera device index or path passed to cv2.VideoCapture (default: 0).",
    )
    parser.add_argument(
        "--shape-name",
        default="shape",
        help="Name of the shared memory block that stores the frame shape.",
    )
    parser.add_argument(
        "--frame-name",
        default="frame",
        help="Name of the shared memory block that stores the frame bytes.",
    )
    parser.add_argument(
        "--convert-rgb",
        action="store_true",
        help="Convert frames from OpenCV's default BGR to RGB before sharing.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Unable to open camera device {args.device}", file=sys.stderr)
        return 1

    shape_segment: Optional[shared_memory.SharedMemory] = None
    frame_segment: Optional[shared_memory.SharedMemory] = None

    def _signal_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if args.convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = frame.astype(FRAME_DTYPE, copy=False)

            shape_array = np.array(frame.shape, dtype=SHAPE_DTYPE)

            if shape_segment is None:
                shape_segment = shared_memory.SharedMemory(
                    create=True, size=shape_array.nbytes, name=args.shape_name
                )
                frame_segment = shared_memory.SharedMemory(
                    create=True, size=frame.nbytes, name=args.frame_name
                )

            shape_segment.buf[: shape_array.nbytes] = shape_array.tobytes()
            frame_segment.buf[: frame.nbytes] = frame.tobytes()
    except KeyboardInterrupt:
        print("Stopping webcam server...")
    finally:
        cap.release()
        _cleanup_shared_memory(frame_segment, shape_segment)

    return 0


if __name__ == "__main__":
    sys.exit(main())
