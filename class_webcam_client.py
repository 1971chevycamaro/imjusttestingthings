"""Shared-memory webcam frame client."""

from __future__ import annotations

from multiprocessing import shared_memory
from typing import Tuple

import numpy as np

SHAPE_DTYPE = np.int32
FRAME_DTYPE = np.uint8


class FrameClient:
    """Client that attaches to shared memory segments and retrieves frames."""

    def __init__(
        self,
        shape_name: str = "shape",
        frame_name: str = "frame",
        *,
        copy: bool = True,
    ) -> None:
        self._shape_segment = shared_memory.SharedMemory(name=shape_name)
        self._frame_segment = shared_memory.SharedMemory(name=frame_name)
        self._copy = copy

    def _read_shape(self) -> Tuple[int, int, int]:
        shape_array = np.ndarray((3,), dtype=SHAPE_DTYPE, buffer=self._shape_segment.buf)
        return tuple(int(v) for v in shape_array)

    def get_frame(self) -> np.ndarray:
        shape = self._read_shape()
        frame_array = np.ndarray(shape, dtype=FRAME_DTYPE, buffer=self._frame_segment.buf)
        if self._copy:
            return frame_array.copy()
        return frame_array

    def close(self) -> None:
        self._frame_segment.close()
        self._shape_segment.close()

    def __enter__(self) -> "FrameClient":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
