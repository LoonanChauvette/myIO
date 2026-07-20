# myIO

Sounddevice `AudioEngine` with a device selector and simple player mixing.

## Install

```bash
uv sync
```

## Quick start

```python
from myio import AudioEngine, Route, select_audio_config

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

engine.listen("space")
engine.listen("escape")
engine.add(my_source, routes=[Route(src=0, dst=0)])
engine.start()
# ...
engine.stop()
```

`select_audio_config(open_ui=False)` requires `config_path`. Cancel in the selector exits the process with status 0.

For PortAudio system defaults with no profile, use `AudioEngine()` / `AudioEngine.default()`.

## Audio sources

Sources declare their number of virtual channels and implement `mix`. They must
**add into** the supplied buffer rather than replace it:

```python
class Tone:
    channels = 1

    def mix(self, buffer, context) -> None:
        buffer += samples  # in-place add
```

Routes connect a source channel to an output channel:

```python
engine.add(tone, routes=[Route(src=0, dst=0, gain=1)])
```

## Keyboard input

Register keyboard keys before starting the engine. Registration does not open
the audio device or create a PsychHID queue:

```python
engine = AudioEngine(samplerate=48000, channels=2)
engine.listen("space")
engine.listen("escape")
```

Players receive key presses and releases through `context.events`:

```python
def mix(self, buffer, context) -> None:
    for event in context.events:
        if event.key == "space" and event.pressed:
            self.trigger(event.sample)
```

Each event contains:

- `key`: normalized key name
- `pressed`: `True` for a press and `False` for a release
- `timestamp`: the original PsychHID/GetSecs timestamp
- `sample`: the absolute engine sample frame mapped to the DAC clock

The sample is not clamped to the current callback. Input is delivered after it
is observed, so its measured sample can be earlier than `context.frame`.
PsychHID, its queue, and the polling thread are started and stopped
automatically with the engine.

To verify the installed PsychHID keycode mapping against a physical keyboard:

```bash
uv run python examples/diagnose_keyboard.py
```

Pass key names as arguments to test a different set.

## Profiles

JSON under a config folder. Profiles store both `device` (PortAudio index) and `device_name`; on load the index is rematched by name under the saved host API and channel count. If nothing matches, `DeviceResolveError` lists eligible devices and a `select_audio_config(..., open_ui=True)` command to fix the profile. Use Save in the UI to write profiles.

## Example

```bash
uv run python examples/play_tones.py
```
