"""Configuration for ``AudioEngine`` / ``sounddevice.OutputStream``."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import sounddevice as sd

Dtype = Literal["float32", "int16", "int32", "uint8"]
Latency = Literal["low", "high"] | float
PathLike = str | Path


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Parameters that map 1:1 to ``sd.OutputStream`` kwargs."""

    samplerate: float
    device: int
    channels: int
    dtype: Dtype = "float32"
    blocksize: int = 0
    latency: Latency = "high"
    clip_off: bool = False
    dither_off: bool = False
    never_drop_input: bool = False
    prime_output_buffers_using_stream_callback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamConfig:
        return cls(
            **{
                f.name: data[f.name]
                for f in fields(cls)
                if f.name in data and data[f.name] is not None
            }
        )


@dataclass(frozen=True, slots=True)
class AudioEngineConfig:
    """Concrete ``StreamConfig`` plus host-API metadata for the selector."""

    stream: StreamConfig
    api: str
    exclusive: bool = False

    @classmethod
    def default(cls) -> AudioEngineConfig:
        """Resolve PortAudio system defaults into a concrete config."""
        with sd.OutputStream() as stream:
            device = stream.device
            if isinstance(device, (tuple, list)):
                device = device[1]
            device = int(device)
            hostapi = int(sd.query_devices(device)["hostapi"])
            return cls(
                stream=StreamConfig(
                    samplerate=float(stream.samplerate),
                    device=device,
                    channels=int(stream.channels),
                ),
                api=str(sd.query_hostapis(hostapi)["name"]),
                exclusive=False,
            )

    def stream_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``sd.OutputStream``."""
        kwargs = self.stream.to_dict()
        if self.exclusive and "WASAPI" in self.api.upper():
            kwargs["extra_settings"] = sd.WasapiSettings(exclusive=True)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "exclusive": self.exclusive,
            "stream": self.stream.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioEngineConfig:
        api = data.get("api")
        if not api:
            raise ValueError("config missing required field: api")
        return cls(
            stream=StreamConfig.from_dict(dict(data.get("stream") or {})),
            api=str(api),
            exclusive=bool(data.get("exclusive", False)),
        )

    def to_file(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: PathLike) -> AudioEngineConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid config file (expected object): {path}")
        return cls.from_dict(data)

    @classmethod
    def from_selector(
        cls,
        config_dir: PathLike | None = None,
        profile: str | None = None,
        parent: Any | None = None,
    ) -> AudioEngineConfig:
        """Show the device selector and return a config.

        Exits the process with status 0 if the user cancels.
        """
        from .selector import DeviceConfigSelector

        return DeviceConfigSelector.select(
            config_dir=config_dir,
            profile=profile,
            parent=parent,
        )


def profile_path(config_dir: PathLike, profile: str) -> Path:
    name = profile.strip()
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    return Path(config_dir) / name


def list_profiles(config_dir: PathLike) -> list[str]:
    path = Path(config_dir)
    if not path.is_dir():
        return []
    return sorted(p.stem for p in path.glob("*.json") if p.is_file())
