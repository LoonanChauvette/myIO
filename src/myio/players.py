from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from myio.engine import AudioContext


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
