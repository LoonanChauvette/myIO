# myIO

Sounddevice `AudioEngine` with a device selector and simple player mixing.

## Install

```bash
uv sync
```

## Quick start

```python
from myio import AudioEngine, select_audio_config

# Configure the audio engine with an UI
config = select_audio_config()

# Or load a saved profile
config = select_audio_config(config_path="audioconfigs/default.json", open_ui=False)

# Or open the UI to a profile from a config folder:
config = select_audio_config(config_folder="audioconfigs", open_ui=True)

# Or pass any valid OutputStream kwargs dict:
config = {"samplerate": 48000, "channels": 2, "latency": "high"}

engine = AudioEngine.from_dict(config)

# Or simply use the default config
engine = AudioEngine.default()

# Or simply pass the arguments like you would to sd.OutputStream
engine = AudioEngine.from_args(samplerate=48000, channels=2, latency="high")

engine.add_player(my_player)
engine.start()
# ...
engine.stop()
```

`select_audio_config(open_ui=False)` requires `config_path`. Cancel in the selector exits the process with status 0.

For PortAudio system defaults with no profile, use `AudioEngine()` / `AudioEngine.default()`.

## Players

```python
# Binds a player to a specific channel
player = Player(channels=0)

# Binds a player to multiple channels
player = Player(channels=[0, 1])

# Binds a player to all available channels
player = Player(channels="all")

# Bind player to multiple groups of channels (e.g. 7.1 surround)
player = Player(channels=[[0, 1], [2, 3, 4, 5, 6, 7]])

# Use a dict to have named channel groups
player = Player(channels={"group1": [1, 2], "group2": [3, 4, 5]})
```

Players must implement `mix` and **add into** the buffer (do not replace it):

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
