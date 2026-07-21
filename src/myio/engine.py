from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Literal, Self, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio.audiosources import AudioContext, AudioSource, CallbackTime
from myio.keyboard import KeyboardQueue, KeyEvent, normalize_key_name

AudioType = Literal["float32", "int32", "int16", "int8", "uint8"]
StreamLatency = Literal["low", "high"]


# Arguments accepted by sounddevice.OutputStream.
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
    render: Callable[[npt.NDArray[np.float32], AudioContext], None]
    channels: int
    buffer: np.ndarray
    routes: list[Route]


@dataclass
class Route:
    src: int = 0
    dst: int = 0
    gain: int = 1


class AudioEngine:
    MAX_BLOCKSIZE: int = 2048

    def __init__(self, **kwargs: Unpack[OutputStreamKwargs]) -> None:
        self._stream_kwargs: OutputStreamKwargs = kwargs
        self._stream: sd.OutputStream | None = None

        self._render_lock: Lock = Lock()  # Protects anything touching the callback
        self._engine_lock: Lock = Lock()
        self._handles: tuple[Handle, ...] = tuple()
        self._frame: int = 0

        self._keys: set[str] = set()
        self._keyboard: KeyboardQueue | None = None

        self.channels: int = int(kwargs.get("channels", 0))
        self.blocksize: int = int(kwargs.get("blocksize", 0))
        self.samplerate: float = float(kwargs.get("samplerate", 0.0))

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
        handle = Handle(
            source=source,
            render=source.mix,
            channels=source.channels,
            buffer=np.empty((self.MAX_BLOCKSIZE, source.channels), dtype=np.float32),
            routes=routes,
        )

        with self._engine_lock:
            # handles are stored as an immutable tuple (better for the render loop)
            # but means we need to reconstruct the tuple on each update
            self._handles = tuple((*self._handles, handle))
        return handle

    def get_handle(self, source: AudioSource) -> Handle:
        """Retrieve the handle associated with an audio source."""
        for handle in self._handles:
            if handle.source is source:
                return handle
        raise ValueError(f"Could not remove sound source {source} : Player not found")

    def remove(self, item: Handle | AudioSource) -> None:
        with self._render_lock:
            if isinstance(item, AudioSource):
                handle = self.get_handle(item)
            elif isinstance(item, Handle):
                handle = item
            self._handles = tuple(h for h in self._handles if h != handle)

    def listen(self, key: str) -> None:
        name = normalize_key_name(key)
        with self._engine_lock:
            if self._stream is not None:
                raise RuntimeError("Cannot register keys while the engine is running")
            self._keys.add(name)

    def start(self) -> None:
        with self._engine_lock:
            if self._stream is not None:
                return

            self._frame = 0
            try:
                self._start_keyboard()
                self._start_stream()
                return
            except Exception as exc:
                stream, self._stream = self._stream, None
                keyboard, self._keyboard = self._keyboard, None
                error = exc

        self._close_stream(stream)
        if keyboard is not None:
            keyboard.stop()
        raise error

    def stop(self) -> None:
        self._cleanup_resources()

    def close(self) -> None:
        self._cleanup_resources()

    def _start_stream(self) -> None:
        stream = sd.OutputStream(callback=self.callback, **self._stream_kwargs)
        self._stream = stream
        self.channels = int(stream.channels)
        self.blocksize = int(stream.blocksize)
        self.samplerate = float(stream.samplerate)
        stream.start()

    def _start_keyboard(self) -> None:
        if not self._keys:
            return

        self._keyboard = KeyboardQueue(self._keys)
        self._keyboard.start()

    def _cleanup_resources(self) -> None:
        with self._engine_lock:
            stream, self._stream = self._stream, None
            keyboard, self._keyboard = self._keyboard, None

        self._close_stream(stream)
        if keyboard is not None:
            keyboard.stop()

    def _close_stream(self, stream: sd.OutputStream | None) -> None:
        if stream is None:
            return
        if stream.active:
            stream.stop()
        stream.close()

    def _drain_pending_events(self, time: CallbackTime) -> tuple[KeyEvent, ...]:
        keyboard = self._keyboard
        if keyboard is None:
            return ()

        if keyboard.error is not None:
            raise RuntimeError("Keyboard worker failed") from keyboard.error

        events: list[KeyEvent] = []
        for event in keyboard.pending_events:
            offset = event.timestamp - time.outputBufferDacTime
            local_sample = offset * self.samplerate
            event.sample = self._frame + int(round(local_sample))
            events.append(event)

        return tuple(events)

    def callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: CallbackTime,
        status: sd.CallbackFlags,
    ) -> None:

        outdata.fill(0)
        context = AudioContext(
            frame=self._frame,
            frames=frames,
            samplerate=self.samplerate,
            time=time,
            status=status,
            events=self._drain_pending_events(time),
        )

        handles = self._handles
        for handle in handles:
            buffer = handle.buffer[:frames]
            buffer.fill(0)
            handle.render(buffer, context)

            for route in handle.routes:
                outdata[:, route.dst] += route.gain * buffer[:, route.src]

        self._frame += frames
