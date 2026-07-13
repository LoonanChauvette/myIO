"""Select a device profile and play a short mixed tone demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from myio import AudioContext, AudioEngine, select_audio_config

CONFIG_DIR = Path(__file__).resolve().parent / "audioconfigs"


class Tone:
    """Simple continuous sine player using the engine's mix() API."""

    def __init__(
        self,
        frequency: float = 440.0,
        amplitude: float = 0.2,
        samplerate: float = 48000.0,
    ) -> None:
        self.frequency = frequency
        self.amplitude = amplitude
        self.samplerate = samplerate
        self._phase = 0.0

    def mix(
        self,
        buffer: npt.NDArray[np.float32],
        context: AudioContext,
    ) -> None:
        frames, channels = buffer.shape
        t = (self._phase + np.arange(frames)) / self.samplerate
        self._phase += frames
        mono = (self.amplitude * np.sin(2 * np.pi * self.frequency * t)).astype(
            np.float32
        )
        buffer += np.repeat(mono.reshape(-1, 1), channels, axis=1)


def main() -> None:
    # config = select_audio_config(config_folder=CONFIG_DIR, open_ui=True)
    config = select_audio_config(config_path=CONFIG_DIR / "basic.json")

    print(f"Using config: {config}")
    engine = AudioEngine.from_dict(config)
    a = Tone(440.0, amplitude=0.25, samplerate=engine.samplerate)
    b = Tone(660.0, amplitude=0.12, samplerate=engine.samplerate)

    engine.add_player(a)
    engine.add_player(b)
    engine.start()

    print(f"Playing @ {engine.samplerate} Hz…")
    try:
        sd.sleep(2000)
        engine.remove_player(b)
        sd.sleep(1000)
    finally:
        engine.stop()
    print("Done.")


if __name__ == "__main__":
    main()
