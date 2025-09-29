"""Display frames broadcast via shared memory."""

from __future__ import annotations

import argparse
import signal
import sys

import cv2

from class_webcam_client import FrameClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--no-copy",
        action="store_true",
        help="Display frames without copying them out of shared memory.",
    )
    args = parser.parse_args()

    def _signal_handler(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        with FrameClient(args.shape_name, args.frame_name, copy=not args.no_copy) as client:
            while True:
                frame = client.get_frame()
                if frame is None:
                    continue

                cv2.imshow("Shared Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except FileNotFoundError:
        print(
            "Shared memory segments not found. Ensure the webcam server is running first.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Stopping webcam client...")
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
