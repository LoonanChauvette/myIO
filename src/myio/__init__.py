"""myIO — sounddevice AudioEngine with device selection and player mixing."""

from myio.config import (
    AudioEngineConfig,
    DeviceResolveError,
    StreamConfig,
    list_profiles,
    profile_path,
)
from myio.engine import AudioEngine, Player, AudioContext
from myio.selector import DeviceConfigSelector
from myio.utils import dbfs_to_rms, rms_to_dbfs


__all__ = [
    "AudioEngine",
    "AudioEngineConfig",
    "AudioContext",
    "DeviceConfigSelector",
    "DeviceResolveError",
    "Player",
    "StreamConfig",
    "list_profiles",
    "profile_path",
    "dbfs_to_rms",
    "rms_to_dbfs",
]
