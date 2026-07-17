from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Literal, Self, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio.audiosources import AudioContext, AudioSource, CallbackTime

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


@dataclass
class Handle:
    source: AudioSource
    render: Callable
    channels: int
    buffer: np.ndarray
    routes: list[Route]


@dataclass
class Route:
    src: int = 0
    dst: int = 0
    gain: int = 1


class AudioEngine:
    def __init__(self, **kwargs: Unpack[OutputStreamKwargs]) -> None:
        self._lock = Lock()
        self._handles: list[Handle] = []
        self._frame: int = 0

        self.MAX_BLOCKSIZE = 2048
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

    def add(self, source: AudioSource, routes: list[Route]) -> Handle:
        # TODO: default routes to all dst channels
        blocksize = self.blocksize if self.blocksize != 0 else self.MAX_BLOCKSIZE
        handle = Handle(
            source=source,
            render=source.mix,
            channels=source.channels,
            buffer=np.empty((blocksize, source.channels), dtype=np.float32),
            routes=routes,
        )

        with self._lock:
            self._handles.append(handle)
        return handle

    def get_handle(self, source: AudioSource) -> Handle:
        for handle in self._handles:
            if handle.source is source:
                return handle
        raise ValueError(f"Could not remove sound source {source} : Player not found")

    def remove(self, item: Handle | AudioSource) -> None:
        with self._lock:
            handle = item if isinstance(item, Handle) else self.get_handle(item)
            self._handles.remove(handle)

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

        outdata.fill(0)  # Empty the output buffer
        context = AudioContext(
            frame=self._frame,
            frames=frames,
            samplerate=self.samplerate,
            time=time,
            status=status,
        )
        with self._lock:
            handles = tuple(self._handles)

        for handle in handles:
            assert handle.buffer.shape[0] >= frames, (
                f"buffer shape {handle.buffer.shape} does not match frames {frames}"
            )
            buffer = handle.buffer[:frames]
            buffer.fill(0)
            handle.render(buffer, context)
            for route in handle.routes:
                outdata[:, route.dst] += route.gain * buffer[:, route.src]

        self._frame += frames
