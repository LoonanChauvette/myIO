"""Select a device profile and play a short mixed tone demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio import AudioEngine, AudioEngineConfig

CONFIG_DIR = Path(__file__).resolve().parent / "audioconfigs"


class Tone:
    """Simple continuous sine player using the engine's mix() API."""

    def __init__(
        self,
        frequency: float = 440.0,
        amplitude: float = 0.2,
        samplerate: int = 48000,
    ) -> None:
        self.frequency = frequency
        self.amplitude = amplitude
        self.samplerate = samplerate
        self._phase = 0.0

    def mix(
        self,
        outdata: npt.NDArray[np.float32],
        time: object,
        status: object,
    ) -> None:
        frames, channels = outdata.shape
        t = (self._phase + np.arange(frames)) / self.samplerate
        self._phase += frames
        mono = (self.amplitude * np.sin(2 * np.pi * self.frequency * t))
        mono = mono.astype(np.float32)
        outdata[:, :] += np.repeat(mono.reshape(-1, 1), channels, axis=1)


def main() -> None:
    #config = AudioEngineConfig.default()
    config = AudioEngineConfig.from_selector(config_dir=CONFIG_DIR)
    #config = AudioEngineConfig.from_file(path=CONFIG_DIR / "default.json")

    print(f"Using config: {config}")
    engine = AudioEngine(config)
    a = Tone(440.0, amplitude=0.25, samplerate=engine.fs)
    b = Tone(660.0, amplitude=0.12, samplerate=engine.fs)

    engine.add_player(a)
    engine.add_player(b)
    engine.start()

    print(f"Playing @ {engine.fs} Hz on device {engine.device}…")
    try:
        sd.sleep(2000)
        engine.remove_player(b)
        sd.sleep(1000)
    finally:
        engine.stop()
    print("Done.")


if __name__ == "__main__":

    main()

