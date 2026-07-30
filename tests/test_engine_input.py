from __future__ import annotations

import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import sounddevice as sd

from myio import AudioContext, AudioEngine, Route
from myio.keyboard import KeyEvent


@dataclass
class FakeTime:
    currentTime: float = 0.0
    inputBufferAdcTime: float = 0.0
    outputBufferDacTime: float = 100.0


class CapturingSource:
    channels = 1

    def __init__(self) -> None:
        self.context: AudioContext | None = None

    def mix(self, buffer: np.ndarray, context: AudioContext, /) -> None:
        self.context = context


class FakeStream:
    def __init__(self, *, callback: object, **kwargs: object) -> None:
        self.callback = callback
        self.channels = kwargs.get("channels", 2)
        self.blocksize = kwargs.get("blocksize", 64)
        self.samplerate = kwargs.get("samplerate", 48_000.0)
        self.active = False
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True


class FakeKeyboard:
    instances: list[FakeKeyboard] = []

    def __init__(self, keys: set[str]) -> None:
        self.keys = keys
        self.started = False
        self.stopped = False
        self._events: list[KeyEvent] = []
        self._error: BaseException | None = None
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    @property
    def pending_events(self) -> Iterator[KeyEvent]:
        while self._events:
            yield self._events.pop(0)

    @property
    def error(self) -> BaseException | None:
        return self._error


class EngineInputTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeKeyboard.instances.clear()

    def test_construction_and_listen_allocate_no_resources(self) -> None:
        with patch("myio.engine.sd.OutputStream") as output_stream:
            engine = AudioEngine(samplerate=48_000, channels=2)
            engine.listen("space")
        output_stream.assert_not_called()

    def test_callback_exposes_absolute_sample_position(self) -> None:
        engine = AudioEngine(samplerate=48_000, channels=1, blocksize=64)
        source = CapturingSource()
        engine.add(source, [Route()])

        keyboard = FakeKeyboard({"space"})
        keyboard._events.append(KeyEvent("space", True, 100.0015, 32))
        engine._keyboard = keyboard

        output = np.empty((64, 1), dtype=np.float32)
        engine.callback(output, 64, FakeTime(), sd.CallbackFlags())

        assert source.context is not None
        self.assertEqual(len(source.context.events), 1)
        event = source.context.events[0]
        self.assertEqual(event.key, "space")
        self.assertEqual(event.sample, 72)
        self.assertEqual(event.timestamp, 100.0015)

    def test_callback_raises_when_keyboard_worker_failed(self) -> None:
        engine = AudioEngine(samplerate=48_000, channels=1, blocksize=64)
        source = CapturingSource()
        engine.add(source, [Route()])

        keyboard = FakeKeyboard({"space"})
        keyboard._error = RuntimeError("hid boom")
        engine._keyboard = keyboard

        output = np.empty((64, 1), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "Keyboard worker failed"):
            engine.callback(output, 64, FakeTime(), sd.CallbackFlags())

    def test_start_and_stop_own_keyboard_and_stream(self) -> None:
        with (
            patch("myio.engine.KeyboardQueue", FakeKeyboard),
            patch("myio.engine.sd.OutputStream", FakeStream),
        ):
            engine = AudioEngine(samplerate=48_000, channels=2, blocksize=64)
            engine.listen("escape")
            engine.start()
            stream = engine._stream
            with self.assertRaises(RuntimeError):
                engine.listen("space")
            engine.stop()

        self.assertIsInstance(stream, FakeStream)
        assert isinstance(stream, FakeStream)
        self.assertTrue(stream.closed)
        self.assertEqual(len(FakeKeyboard.instances), 1)
        keyboard = FakeKeyboard.instances[0]
        self.assertEqual(keyboard.keys, {"escape"})
        self.assertTrue(keyboard.started)
        self.assertTrue(keyboard.stopped)

    def test_audio_only_start_does_not_create_keyboard(self) -> None:
        with (
            patch("myio.engine.KeyboardQueue", FakeKeyboard),
            patch("myio.engine.sd.OutputStream", FakeStream),
        ):
            engine = AudioEngine(samplerate=48_000, channels=2, blocksize=64)
            engine.start()
            engine.stop()

        self.assertEqual(FakeKeyboard.instances, [])

    def test_audio_start_failure_releases_keyboard(self) -> None:
        with (
            patch("myio.engine.KeyboardQueue", FakeKeyboard),
            patch(
                "myio.engine.sd.OutputStream",
                side_effect=RuntimeError("audio unavailable"),
            ),
        ):
            engine = AudioEngine(samplerate=48_000, channels=2)
            engine.listen("space")
            with self.assertRaisesRegex(RuntimeError, "audio unavailable"):
                engine.start()

        self.assertIsNone(engine._stream)
        self.assertTrue(FakeKeyboard.instances[0].stopped)


if __name__ == "__main__":
    unittest.main()
