"""Real-time audio output engine.

``AudioEngine`` owns an ``sd.OutputStream`` and mixes registered ``Player``s
each audio block.
"""

from __future__ import annotations

import threading

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from .config import AudioEngineConfig
from .players import Player, _Status, _Time


class AudioEngine:
    """Owns a PortAudio output stream and mixes registered players.

    Pass a concrete ``AudioEngineConfig``, or ``None`` to use
    ``AudioEngineConfig.default()``.
    """

    def __init__(self, config: AudioEngineConfig | None = None) -> None:
        self.config = config or AudioEngineConfig.default()
        self.api = self.config.api
        self.exclusive = self.config.exclusive
        self._lock = threading.Lock()
        self._players: tuple[Player, ...] = ()
        self._output_clipping = False
        self.stream: sd.OutputStream | None = None

        s = self.config.stream
        self.fs = int(s.samplerate)
        self.channels = int(s.channels)
        self.device = int(s.device)
        self.blocksize = s.blocksize

    def _open_stream(self) -> None:
        if self.stream is not None:
            return
        self.stream = sd.OutputStream(
            callback=self._callback,
            **self.config.stream_kwargs(),
        )

    def DAC_time(self) -> float:
        return 0.0 if self.stream is None else self.stream.time

    def add_player(self, player: Player) -> None:
        with self._lock:
            self._players = (*self._players, player)

    def remove_player(self, player: Player) -> None:
        with self._lock:
            self._players = tuple(p for p in self._players if p is not player)

    def _callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: _Time,
        status: _Status,
    ) -> None:
        mix = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
        with self._lock:
            players = self._players
        for player in players:
            player.mix(mix, time, status)
        outdata[:] = mix

        peak = float(np.abs(mix).max()) if frames else 0.0
        if peak > 1.0:
            if not self._output_clipping:
                self._output_clipping = True
                print(f"warning: output clipping (peak {peak:.3f})")
        else:
            self._output_clipping = False

    def start(self) -> None:
        self._open_stream()
        assert self.stream is not None
        if not self.stream.active:
            self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return
        if self.stream.active:
            self.stream.stop()
        self.stream.close()
        self.stream = None
