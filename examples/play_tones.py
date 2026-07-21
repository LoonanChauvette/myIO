"""Select a device profile and play a short mixed tone demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio import AudioContext, AudioEngine, AudioSource, Route, select_audio_config

CONFIG_DIR = Path(__file__).resolve().parent / "audioconfigs"
TAU = 2 * np.pi


@dataclass
class Tone(AudioSource):
    """Simple continuous sine player using the engine's mix() API."""

    frequency: float = 440.0
    amplitude: float = 0.2
    channels: int = 1
    _phase: float = field(default=0.0, init=False)
    _phase_increment: float | None = field(default=None, init=False)
    _playing: bool = field(default=True, init=False)

    def mix(self, buffer: npt.NDArray[np.float32], ctx: AudioContext) -> None:
        if buffer.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {buffer.shape[1]}"
            )

        for event in ctx.events:
           if event.pressed and event.key == "space":
               self._playing = not self._playing
           elif event.pressed and event.key == "escape":
               self._playing = False

        if not self._playing:
            return

        if self._phase_increment is None:
            self._phase_increment = TAU * self.frequency / ctx.samplerate

        samples = np.arange(ctx.frames, dtype=np.float32)
        samples *= self._phase_increment
        samples += self._phase
        np.sin(samples, out=samples)
        samples *= self.amplitude

        buffer += samples[:, None]

        self._phase += ctx.frames * self._phase_increment
        self._phase %= TAU


def main() -> None:
    config = select_audio_config(config_path=CONFIG_DIR / "basic.json", open_ui=False)
    engine = AudioEngine.from_dict(config)
    engine.listen("space")
    engine.listen("escape")

    a = Tone(1000.0, amplitude=0.24)
    b = Tone(1100.0, amplitude=0.12)

    engine.add(a, routes=[Route(src=0, dst=0)])
    engine.add(b, routes=[Route(src=0, dst=1)])
    engine.start()

    print(f"Playing @ {engine.samplerate} Hz…")
    try:
        sd.sleep(2000)
        engine.remove(b)
        sd.sleep(1000)
    finally:
        print("stopping engine")
        engine.stop()
        print("engine stopped")
    print("Done.")


if __name__ == "__main__":
    main()
