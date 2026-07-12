"""Player protocol for ``AudioEngine`` mixing."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

# sounddevice callback time info (cffi) and status flags.
_Time = Any
_Status = sd.CallbackFlags


@runtime_checkable
class Player(Protocol):
    """Audio source mixed in-place by ``AudioEngine``.

    Implementations should add into ``outdata`` (do not replace it).
    """

    def mix(
        self,
        outdata: npt.NDArray[np.float32],
        time: _Time,
        status: _Status,
    ) -> None: ...
