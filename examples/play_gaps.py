"""Continuity-illusion demo: a tone with noise-filled gaps.

Up/Down adjust SNR, Left/Right adjust segment duration, Space toggles
continuous tone (noise unchanged), Escape quits.

Tone-on and noise-on segments share the same duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt

from myio import AudioContext, AudioEngine, AudioSource, Route, select_audio_config

CONFIG_DIR = Path(__file__).resolve().parent / "audioconfigs"
TAU = 2 * np.pi

SNR_STEP_DB = 1.0
SEGMENT_STEP_S = 0.01
SEGMENT_MIN_S = 0.01


@dataclass
class GappedTone(AudioSource):
    """Alternating equal-length tone and noise segments.

    Within each period the first half is noise (tone off) and the second half
    is tone (noise off). Both halves use ``segment_duration``.
    """

    frequency: float = 1000.0
    amplitude: float = 0.24
    channels: int = 1

    segment_duration: float = 0.25
    snr_db: float = 10.0
    ramp_duration: float = 0.005
    continuous: bool = False

    _phase: float = field(default=0.0, init=False)
    _phase_increment: float | None = field(default=None, init=False)
    _playing: bool = field(default=True, init=False)
    _status_dirty: bool = field(default=True, init=False)

    _rng: np.random.Generator = field(
        default_factory=np.random.default_rng,
        init=False,
    )
    _noise_sos: np.ndarray | None = field(default=None, init=False)
    _noise_state: np.ndarray | None = field(default=None, init=False)

    def mix(
        self,
        buffer: npt.NDArray[np.float32],
        ctx: AudioContext,
    ) -> None:
        if buffer.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {buffer.shape[1]}"
            )

        self._handle_events(ctx)
        if not self._playing:
            return

        if self._phase_increment is None:
            self._init_dsp(ctx.samplerate)

        frames = ctx.frames
        sr = ctx.samplerate
        assert self._phase_increment is not None
        assert self._noise_sos is not None
        assert self._noise_state is not None

        # ---- Tone ----
        phase = (
            np.arange(frames, dtype=np.float64) * self._phase_increment + self._phase
        )
        tone = self.amplitude * np.sin(phase)
        self._phase = float((phase[-1] + self._phase_increment) % TAU)

        # ---- Equal noise / tone windows ----
        segment_samples = max(1, int(round(self.segment_duration * sr)))
        ramp_samples = max(0, int(round(self.ramp_duration * sr)))
        ramp_samples = min(ramp_samples, segment_samples // 2)

        period_samples = 2 * segment_samples
        sample_index = (ctx.frame + np.arange(frames)) % period_samples

        # First half: noise (tone off). Second half: tone (noise off).
        in_noise = sample_index < segment_samples
        in_gap = in_noise

        # ---- Tone envelope ----
        if self.continuous:
            envelope = np.ones(frames, dtype=np.float64)
        else:
            envelope = np.ones(frames, dtype=np.float64)
            envelope[in_gap] = 0.0
            if ramp_samples > 0:
                after_gap = sample_index - segment_samples
                fade_in = (after_gap >= 0) & (after_gap < ramp_samples)
                envelope[fade_in] = after_gap[fade_in] / ramp_samples

                before_gap = period_samples - sample_index
                fade_out = (~in_gap) & (before_gap <= ramp_samples)
                envelope[fade_out] = before_gap[fade_out] / ramp_samples

        # ---- Bandpass noise (always filtered to keep SOS state continuous) ----
        noise = self._rng.standard_normal(frames)
        noise, self._noise_state = sosfilt(self._noise_sos, noise, zi=self._noise_state)

        rms = float(np.sqrt(np.mean(noise**2)))
        if rms > 0:
            tone_rms = self.amplitude / np.sqrt(2)
            noise_rms = tone_rms / (10 ** (self.snr_db / 20))
            noise *= noise_rms / rms

        output = tone * envelope
        output[in_noise] += noise[in_noise]
        buffer += output.astype(np.float32)[:, None]

    def _handle_events(self, ctx: AudioContext) -> None:
        for event in ctx.events:
            if not event.pressed:
                continue

            if event.key == "up":
                self.snr_db += SNR_STEP_DB
                self._status_dirty = True
            elif event.key == "down":
                self.snr_db -= SNR_STEP_DB
                self._status_dirty = True
            elif event.key == "right":
                self.segment_duration += SEGMENT_STEP_S
                self._status_dirty = True
            elif event.key == "left":
                self.segment_duration = max(
                    SEGMENT_MIN_S, self.segment_duration - SEGMENT_STEP_S
                )
                self._status_dirty = True
            elif event.key == "space":
                self.continuous = not self.continuous
                self._status_dirty = True
            elif event.key == "escape":
                self._playing = False

    def _init_dsp(self, samplerate: float) -> None:
        self._phase_increment = TAU * self.frequency / samplerate

        # Two-octave band centered on the tone: f/√2 … f·√2
        factor = np.sqrt(2)
        low = self.frequency / factor
        high = min(self.frequency * factor, samplerate * 0.5 * 0.99)

        self._noise_sos = butter(
            N=4,
            Wn=[low, high],
            btype="bandpass",
            fs=samplerate,
            output="sos",
        )
        self._noise_state = np.zeros((self._noise_sos.shape[0], 2), dtype=np.float64)

    def status_line(self) -> str:
        mode = "continuous" if self.continuous else "gapped"
        return (
            f"SNR={self.snr_db:+.0f} dB  "
            f"segment={self.segment_duration * 1000:.0f} ms  "
            f"tone={mode}"
        )


def main() -> None:
    config = select_audio_config(config_path=CONFIG_DIR / "basic.json", open_ui=False)
    engine = AudioEngine.from_dict(config)

    for key in ("up", "down", "left", "right", "space", "escape"):
        engine.listen(key)

    stimulus = GappedTone(
        frequency=1000.0,
        amplitude=0.24,
        segment_duration=0.25,
        snr_db=10.0,
    )
    engine.add(
        stimulus,
        routes=[
            Route(src=0, dst=0),
            Route(src=0, dst=1),
        ],
    )
    engine.start()

    print(f"Playing @ {engine.samplerate} Hz…")
    print(
        "Controls: Up/Down SNR  Left/Right segment  Space continuous  Escape quit"
    )
    print(stimulus.status_line())
    stimulus._status_dirty = False

    try:
        while stimulus._playing:
            if stimulus._status_dirty:
                print(stimulus.status_line())
                stimulus._status_dirty = False
            sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    print("Done.")


if __name__ == "__main__":
    main()
