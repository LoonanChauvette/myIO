from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sounddevice as sd


class CallbackTime(Protocol):
    currentTime: float
    inputBufferAdcTime: float
    outputBufferDacTime: float


@dataclass(frozen=True)
class AudioContext:
    frame: int
    frames: int
    samplerate: float
    time: CallbackTime
    status: sd.CallbackFlags


class Player(Protocol):
    """Players should implement ``mix`` by adding their audio output into the
    provided buffer. The buffer is shared with other players, so it should
    not be replaced.

    Example::

        class SinePlayer:
            def mix(self, buffer, context) -> None:
                samples = generate_audio(context.frames, context.samplerate)
                buffer += samples
    """

    def mix(
        self,
        buffer: npt.NDArray[np.float32],
        context: AudioContext,
    ) -> None: ...
