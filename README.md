# myIO

Sounddevice `AudioEngine` with a device selector and simple player mixing.

## Install

```bash
uv sync
```

## Quick start

```python
from myio import AudioEngine, select_audio_config

# Load a saved profile (flat OutputStream kwargs):
config = select_audio_config(config_path="audioconfigs/default.json")

# Or open the UI to pick / save a device:
# config = select_audio_config(config_folder="audioconfigs", open_ui=True)

# Or pass any valid OutputStream kwargs dict:
# config = {"samplerate": 48000, "channels": 2, "latency": "high"}

engine = AudioEngine.from_dict(config)
engine.add_player(my_player)
engine.start()
# ...
engine.stop()
```

`select_audio_config(open_ui=False)` requires `config_path`. Cancel in the selector exits the process with status 0.

For PortAudio system defaults with no profile, use `AudioEngine()` / `AudioEngine.default()`.

## Players

Implement `mix` and **add into** the buffer (do not replace it):

```python
class Tone:
    def mix(self, buffer, context) -> None:
        buffer += samples  # in-place add
```

## Profiles

JSON under a config folder. Profiles store both `device` (PortAudio index) and `device_name`; on load the index is rematched by name under the saved host API and channel count. If nothing matches, `DeviceResolveError` lists eligible devices and a `select_audio_config(..., open_ui=True)` command to fix the profile. Use Save in the UI to write profiles.

## Example

```bash
uv run python examples/play_tones.py
```
