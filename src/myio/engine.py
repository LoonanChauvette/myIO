from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Literal, Protocol, TypedDict, Unpack

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
    samplerate: int
    time: CallbackTime
    status: sd.CallbackFlags


AudioType = Literal["float32", "int32", "int16", "int8", "uint8"]
StreamLatency = Literal["low", "high"]


class OutputStreamKwargs(TypedDict, total=False):
    samplerate: float
    blocksize: int
    device: int | str
    channels: int
    dtype: AudioType | np.dtype
    latency: StreamLatency | float
    extra_settings: sd.AsioSettings | sd.CoreAudioSettings | sd.WasapiSettings
    clip_off: bool
    dither_off: bool
    never_drop_input: bool
    prime_output_buffers_using_stream_callback: bool


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

        self.stream = sd.OutputStream(callback=self.callback, **kwargs)
        self.channels = self.stream.channels
        self.blocksize = self.stream.blocksize
        self.samplerate = self.stream.samplerate

    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def from_args(cls, **kwargs: Unpack[OutputStreamKwargs]):
        return cls(**kwargs)

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
