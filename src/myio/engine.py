from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio.players import Player

AudioType = Literal["float32", "int32", "int16", "int8", "uint8"]
StreamLatency = Literal["low", "high"]


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


class AudioEngine:
    def __init__(self, **kwargs) -> None:
        self._lock = Lock()
        self._players: list[Player] = []
        self._frame: int = 0

        self.stream = sd.OutputStream(callback=self.callback, **kwargs)
        self.channels: int = self.stream.channels
        self.blocksize: int = self.stream.blocksize
        self.samplerate: int = self.stream.samplerate

    @classmethod
    def default(cls) -> AudioEngine:
        return cls()

    @classmethod
    def from_args(cls, **kwargs: Unpack[OutputStreamKwargs]) -> AudioEngine:
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, config: dict) -> AudioEngine:
        return cls(**config)

    # TODO: hook up from_file and from selector

    def add_player(self, player: Player) -> None:
        with self._lock:
            self._players.append(player)

    def remove_player(self, player: Player) -> None:
        with self._lock:
            self._players.remove(player)

    def start(self) -> None:
        if not self.stream.active:
            self.stream.start()

    def stop(self) -> None:
        if self.stream.active:
            self.stream.stop()

    def close(self) -> None:
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
