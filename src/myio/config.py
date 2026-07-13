"""Configuration for ``AudioEngine`` / ``sounddevice.OutputStream``."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import sounddevice as sd

Dtype = Literal["float32", "int16", "int32", "uint8"]
Latency = Literal["low", "high"] | float
PathLike = str | Path


class DeviceResolveError(LookupError):
    """Raised when a saved device cannot be rematched on this system."""


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
    """Concrete ``StreamConfig`` plus host-API / device identity for profiles."""

    stream: StreamConfig
    api: str
    device_name: str
    exclusive: bool = False

    @classmethod
    def default(cls) -> AudioEngineConfig:
        """Resolve PortAudio system defaults into a concrete config."""
        with sd.OutputStream() as stream:
            device = stream.device
            if isinstance(device, (tuple, list)):
                device = device[1]
            device = int(device)
            info = sd.query_devices(device)
            hostapi = int(info["hostapi"])
            return cls(
                stream=StreamConfig(
                    samplerate=float(stream.samplerate),
                    device=device,
                    channels=int(stream.channels),
                ),
                api=str(sd.query_hostapis(hostapi)["name"]),
                device_name=str(info["name"]),
                exclusive=False,
            )

    def resolve(self, *, config_path: PathLike | None = None) -> AudioEngineConfig:
        """Rematch ``stream.device`` using ``device_name``, ``api``, and channels.

        When ``device_name`` is set, a device with that exact name must exist on
        the saved host API with at least ``stream.channels`` outputs. The saved
        index is kept if it still points at that name; otherwise it is updated.

        When ``device_name`` is empty (legacy profiles), the saved index is used
        if it remains valid for the API/channels, and ``device_name`` is filled in.

        Raises ``DeviceResolveError`` when nothing matches.
        """
        channels = int(self.stream.channels)
        api_id = next(
            (
                i
                for i, host in enumerate(sd.query_hostapis())
                if host["name"] == self.api
            ),
            None,
        )
        if api_id is None:
            apis = ", ".join(repr(h["name"]) for h in sd.query_hostapis()) or "(none)"
            raise DeviceResolveError(
                f"Host API {self.api!r} is not available on this system.\n"
                f"Available host APIs: {apis}\n"
                f"{_selector_hint(config_path)}"
            )

        eligible: list[tuple[int, str, int]] = []
        for i, dev in enumerate(sd.query_devices()):
            max_ch = int(dev["max_output_channels"])
            if max_ch < channels:
                continue
            if int(dev["hostapi"]) != api_id:
                continue
            eligible.append((i, str(dev["name"]), max_ch))

        if self.device_name:
            by_name = [
                (i, name, max_ch)
                for i, name, max_ch in eligible
                if name == self.device_name
            ]
            if self.stream.device in {i for i, _, _ in by_name}:
                return self
            if by_name:
                return replace(self, stream=replace(self.stream, device=by_name[0][0]))
        else:
            for i, name, _max_ch in eligible:
                if i == self.stream.device:
                    return replace(self, device_name=name)

        raise DeviceResolveError(
            _device_resolve_message(
                api=self.api,
                device_name=self.device_name,
                device=self.stream.device,
                channels=channels,
                eligible=eligible,
                config_path=config_path,
            )
        )

    def stream_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``sd.OutputStream`` (device index refreshed via ``resolve``)."""
        cfg = self.resolve()
        kwargs = cfg.stream.to_dict()
        if cfg.exclusive and "WASAPI" in cfg.api.upper():
            kwargs["extra_settings"] = sd.WasapiSettings(exclusive=True)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "device_name": self.device_name,
            "exclusive": self.exclusive,
            "stream": self.stream.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        config_path: PathLike | None = None,
    ) -> AudioEngineConfig:
        api = data.get("api")
        if not api:
            raise ValueError("config missing required field: api")
        stream = StreamConfig.from_dict(dict(data.get("stream") or {}))
        device_name = data.get("device_name")
        if not device_name:
            try:
                device_name = str(sd.query_devices(stream.device)["name"])
            except Exception:
                device_name = ""
        return cls(
            stream=stream,
            api=str(api),
            device_name=str(device_name),
            exclusive=bool(data.get("exclusive", False)),
        ).resolve(config_path=config_path)

    def to_file(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: PathLike) -> AudioEngineConfig:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid config file (expected object): {path}")
        return cls.from_dict(data, config_path=path)

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


def _selector_hint(config_path: PathLike | None) -> str:
    if config_path is not None:
        path = Path(config_path)
        config_dir = path.parent.as_posix()
        profile = path.stem
        return (
            "Open the selector to pick a valid device and Save the profile, e.g.\n"
            "  python -c \"from myio import AudioEngineConfig; "
            f"AudioEngineConfig.from_selector(config_dir=r'{config_dir}', "
            f"profile='{profile}')\""
        )
    return (
        "Open the selector to pick a valid device and Save the profile, e.g.\n"
        "  python -c \"from myio import AudioEngineConfig; "
        "AudioEngineConfig.from_selector(config_dir='YOUR_CONFIG_DIR', "
        "profile='YOUR_PROFILE')\""
    )


def _device_resolve_message(
    *,
    api: str,
    device_name: str,
    device: int,
    channels: int,
    eligible: list[tuple[int, str, int]],
    config_path: PathLike | None,
) -> str:
    wanted = (
        f"device_name={device_name!r}, device={device}, "
        f"api={api!r}, channels>={channels}"
    )
    if eligible:
        lines = "\n".join(
            f"  [{i}] {name}  (max_output_channels={max_ch})"
            for i, name, max_ch in eligible
        )
        available = (
            f"Output devices on {api!r} with at least {channels} channel(s):\n{lines}"
        )
    else:
        available = (
            f"No output devices on {api!r} provide at least {channels} channel(s)."
        )
    return (
        f"Could not resolve audio device ({wanted}).\n"
        f"{available}\n"
        f"{_selector_hint(config_path)}"
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
