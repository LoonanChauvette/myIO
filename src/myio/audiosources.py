from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio.clock import CallbackTime, Clock
from myio.keyboard import KeyEvent


@dataclass(frozen=True, slots=True)
class AudioContext:
    frames: int
    samplerate: float
    clock: Clock
    events: tuple[KeyEvent, ...] = ()


@runtime_checkable
class AudioSource(Protocol):
    """Audio sources should implement ``mix`` by adding their audio output into the
    provided buffer. The buffer is shared with other sources, so it should
    not be replaced.

    Example::

        class SinePlayer:
            def mix(self, buffer, context) -> None:
                samples = generate_audio(context.frames, context.samplerate)
                buffer += samples
    """

    channels: int

    def mix(
        self,
        buffer: npt.NDArray[np.float32],
        context: AudioContext,
        /,  # means cannot be passed as keyword arguments
    ) -> None: ...
