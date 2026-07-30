from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Self, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio.audiosources import AudioContext, AudioSource
from myio.clock import CallbackTime, Clock
from myio.events import BaseEvent, CallbackEvent, EventQueue
from myio.keyboard import KeyboardQueue, normalize_key_name

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

    @classmethod
    def from_source(cls, source: AudioSource, max: int, routes: list[Route]) -> Self:
        return cls(
            source=source,
            render=source.mix,
            channels=source.channels,
            buffer=np.empty((max, source.channels), dtype=np.float32),
            routes=routes,
        )


@dataclass
class Route:
    src: int = 0
    dst: int = 0
    gain: int = 1


class AudioEngine:
    MAX_BLOCKSIZE: int = 2048

    def __init__(self, **kwargs: Unpack[OutputStreamKwargs]) -> None:
        self.stream: sd.OutputStream = self.init_stream(kwargs)
        self.channels: int = int(self.stream.channels)
        self.blocksize: int = int(self.stream.blocksize)
        self.samplerate: float = float(self.stream.samplerate)

        print(
            f"Stream started with {self.channels} channels, {self.blocksize} blocksize, {self.samplerate} samplerate"
        )

        self.clock: Clock = Clock(self.samplerate, self.blocksize)
        self.event_queue: EventQueue = EventQueue()
        self.keyboard: KeyboardQueue = KeyboardQueue(self.event_queue)

        self._engine_lock: Lock = Lock()
        self._handles: tuple[Handle, ...] = ()

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

    def init_stream(self, kwargs: OutputStreamKwargs) -> sd.OutputStream:
        try:
            stream = sd.OutputStream(callback=self.callback, **kwargs)
            return stream
        except Exception as e:
            raise RuntimeError("Failed to start stream") from e

    def add(self, source: AudioSource, routes: list[Route]) -> Handle:
        # TODO: default routes to all dst channels
        handle = Handle.from_source(source, self.MAX_BLOCKSIZE, routes)

        with self._engine_lock:
            # handles are stored as an immutable tuple (better for the render loop)
            # but means we need to reconstruct the tuple on each update
            self._handles = (*self._handles, handle)
        return handle

    def get_handle(self, source: AudioSource) -> Handle:
        """Retrieve the handle associated with an audio source."""
        for handle in self._handles:
            if handle.source is source:
                return handle
        raise ValueError(f"Could not remove sound source {source} : Player not found")

    def remove(self, item: Handle | AudioSource) -> None:
        with self._engine_lock:
            if isinstance(item, AudioSource):
                handle = self.get_handle(item)
            elif isinstance(item, Handle):
                handle = item
            self._handles = tuple(h for h in self._handles if h != handle)

    def listen(self, key: str) -> None:
        name = normalize_key_name(key)
        with self._engine_lock:
            if self.stream.active:
                raise RuntimeError("Cannot register keys while the engine is running")
            try:
                self.keyboard.add_keys({name})
            except RuntimeError as e:
                raise RuntimeError(f"Failed to register key {name}") from e

    def start(self) -> None:
        try:
            with self._engine_lock:
                self.keyboard.start()
                self.stream.start()
                return
        except Exception as exc:
            self.stop()
            raise RuntimeError("Failed to start the engine") from exc

    def stop(self) -> None:
        with self._engine_lock:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.keyboard.stop()

            for _ in self.event_queue.pending_events:
                pass

    def _collect_events(self, clock: Clock) -> list[BaseEvent]:
        event_queue = self.event_queue
        keyboard = self.keyboard
        if keyboard.error is not None:
            raise RuntimeError("Keyboard worker failed") from keyboard.error

        events = []
        for event in event_queue.pending_events:
            if event.sample is None:
                offset = event.timestamp - clock.output_time
                local_sample = offset * self.samplerate
                event.sample = clock.frame + round(local_sample)
            events.append(event)
        return events

    def render(self, outdata: npt.NDArray[np.float32], ctx: AudioContext) -> None:
        outdata.fill(0)
        handles = self._handles
        for handle in handles:
            buffer = handle.buffer[: ctx.frames]
            buffer.fill(0)
            handle.render(buffer, ctx)
            for route in handle.routes:
                outdata[:, route.dst] += route.gain * buffer[:, route.src]

    def callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: CallbackTime,
        status: sd.CallbackFlags,
    ) -> None:
        clock = self.clock
        clock.tick(frames, time)

        events = self._collect_events(clock)
        if status:
            events.append(
                CallbackEvent.from_flags(status, clock.current_time, clock.frame)
            )

        self.render(outdata, AudioContext(frames, self.samplerate, clock, events))
