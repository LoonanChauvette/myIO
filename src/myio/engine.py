from __future__ import annotations

from threading import Lock
from typing import Literal, Self, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio.players import AudioContext, CallbackTime, Player

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


class AudioEngine:
    def __init__(self, **kwargs: Unpack[OutputStreamKwargs]) -> None:
        self._lock = Lock()
        self._players: list[Player] = []
        self._frame: int = 0

        self.stream = sd.OutputStream(callback=self.callback, **kwargs)
        self.channels: int = self.stream.channels
        self.blocksize: int = self.stream.blocksize
        self.samplerate: float = self.stream.samplerate

    @classmethod
    def default(cls) -> Self:
        return cls()

    @classmethod
    def from_args(cls, **kwargs: Unpack[OutputStreamKwargs]) -> Self:
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, config: dict | OutputStreamKwargs) -> Self:
        # TODO: validate config with OutputStreamKwargs
        return cls(**config)

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
        for player in players:
            player.mix(buffer, context)

        outdata[:] = buffer
        self._frame = frame + frames
