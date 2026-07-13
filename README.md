# myIO

Sounddevice `AudioEngine` with a device selector and simple player mixing.

## Install

```bash
uv sync
```

## Quick start

```python
from myio import AudioEngine, AudioEngineConfig

# Pick one:
config = AudioEngineConfig.from_selector(config_dir="audioconfigs", profile="default")
# config = AudioEngineConfig.from_file("audioconfigs/default.json")
# config = AudioEngineConfig.default()  # PortAudio system defaults

engine = AudioEngine(config)  # or AudioEngine() → default()
engine.add_player(my_player)
engine.start()
# ...
engine.stop()
```

Cancel in the selector exits the process with status 0 (no return value to check).

## Players

Implement `mix` and **add into** the buffer (do not replace it):

```python
class Tone:
    def mix(self, outdata, time, status) -> None:
        outdata[:, :] += samples  # in-place add
```

## Profiles

JSON under a config folder. Profiles store both `device` (PortAudio index) and `device_name`; on load the index is rematched by name under the saved host API and channel count. If nothing matches, `DeviceResolveError` lists eligible devices and a `from_selector(...)` command to fix the profile. `from_selector(config_dir=..., profile=...)` opens the UI; use Save to write profiles.

## Example

```bash
uv run python examples/play_tones.py
```
