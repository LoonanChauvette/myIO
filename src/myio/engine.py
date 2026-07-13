
import sounddevice as sd
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from typing import Protocol
from threading import Lock

@dataclass(frozen=True)
class AudioContext:
    frame: int
    frames: int
    samplerate: int
    time: CallbackTime
    status: sd.CallbackFlags

class CallbackTime(Protocol):
    currentTime: float
    inputBufferAdcTime: float
    outputBufferDacTime: float

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

class AudioEngine:
    def __init__(self, **kwargs):
        self._lock = Lock()
        self._players: list[Player] = []
        self._frame: int = 0

        self.stream = sd.OutputStream(
            callback=self.callback,
            **kwargs
        )
        self.channels = self.stream.channels
        self.blocksize = self.stream.blocksize
        self.samplerate = self.stream.samplerate


    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    # TODO: hook up from_file and from selector

    def add_player(self, player: Player):
        with self._lock:
            self._players.append(player)

    def remove_player(self, player: Player):
        with self._lock:
            self._players.remove(player)

    def start(self):
        if not self.stream.active:
            self.stream.start()

    def stop(self):
        if self.stream.active:
            self.stream.stop()

    def close(self):
        if self.stream.active:
            self.stream.stop()
        self.stream.close()

    def callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: CallbackTime,
        status: sd.CallbackFlags,
    ) -> None:

        # Allocation inside the loop ? Probably fine...
        buffer = np.zeros_like(outdata)

        frame = self._frame
        context = AudioContext(
            frame=frame,
            frames=frames,
            samplerate=self.samplerate,
            time=time,
            status=status,
        )
        with self._lock:
            players = tuple(self._players)
        try:
            for player in players:
                player.mix(buffer, context)
        except Exception:
            # TODO: log exception outside audio thread
            outdata.fill(0)

        outdata[:] = buffer
        self._frame = frame + frames
