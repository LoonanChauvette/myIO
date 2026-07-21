from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Thread
from time import sleep
from typing import Iterator, TypedDict

import numpy as np
import numpy.typing as npt
from psychtoolbox import GetSecs, PsychHID


def key_codes(*values: int) -> tuple[int, ...]:
    return values


# On Windows, PsychHID maps DirectInput scan codes to this VK namespace.
WINDOWS_KEY_CODES: dict[str, tuple[int, ...]] = {
    "backspace": key_codes(0x08),
    "tab": key_codes(0x09),
    "clear": key_codes(0x0C),
    "enter": key_codes(0x0D),
    "shift": key_codes(0xA0, 0xA1),
    "control": key_codes(0xA2, 0xA3),
    "alt": key_codes(0xA4, 0xA5),
    "pause": key_codes(0x13),
    "caps_lock": key_codes(0x14),
    "escape": key_codes(0x1B),
    "space": key_codes(0x20),
    "page_up": key_codes(0x21),
    "page_down": key_codes(0x22),
    "end": key_codes(0x23),
    "home": key_codes(0x24),
    "left": key_codes(0x25),
    "up": key_codes(0x26),
    "right": key_codes(0x27),
    "down": key_codes(0x28),
    "select": key_codes(0x29),
    "print": key_codes(0x2A),
    "execute": key_codes(0x2B),
    "print_screen": key_codes(0x2C),
    "insert": key_codes(0x2D),
    "delete": key_codes(0x2E),
    "help": key_codes(0x2F),
    "left_windows": key_codes(0x5B),
    "right_windows": key_codes(0x5C),
    "menu": key_codes(0x5D),
    "sleep": key_codes(0x5F),
    "numpad_multiply": key_codes(0x6A),
    "numpad_add": key_codes(0x6B),
    "numpad_separator": key_codes(0x6C),
    "numpad_subtract": key_codes(0x6D),
    "numpad_decimal": key_codes(0x6E),
    "numpad_divide": key_codes(0x6F),
    "num_lock": key_codes(0x90),
    "scroll_lock": key_codes(0x91),
    "left_shift": key_codes(0xA0),
    "right_shift": key_codes(0xA1),
    "left_control": key_codes(0xA2),
    "right_control": key_codes(0xA3),
    "left_alt": key_codes(0xA4),
    "right_alt": key_codes(0xA5),
    "browser_back": key_codes(0xA6),
    "browser_forward": key_codes(0xA7),
    "browser_refresh": key_codes(0xA8),
    "browser_stop": key_codes(0xA9),
    "browser_search": key_codes(0xAA),
    "browser_favorites": key_codes(0xAB),
    "browser_home": key_codes(0xAC),
    "volume_mute": key_codes(0xAD),
    "volume_down": key_codes(0xAE),
    "volume_up": key_codes(0xAF),
    "media_next": key_codes(0xB0),
    "media_previous": key_codes(0xB1),
    "media_stop": key_codes(0xB2),
    "media_play_pause": key_codes(0xB3),
    "launch_mail": key_codes(0xB4),
    "launch_media": key_codes(0xB5),
    "launch_app_1": key_codes(0xB6),
    "launch_app_2": key_codes(0xB7),
    "semicolon": key_codes(0xBA),
    "equals": key_codes(0xBB),
    "comma": key_codes(0xBC),
    "minus": key_codes(0xBD),
    "period": key_codes(0xBE),
    "slash": key_codes(0xBF),
    "backtick": key_codes(0xC0),
    "left_bracket": key_codes(0xDB),
    "backslash": key_codes(0xDC),
    "right_bracket": key_codes(0xDD),
    "quote": key_codes(0xDE),
    "oem_102": key_codes(0xE2),
}
WINDOWS_KEY_CODES.update(
    {str(number): key_codes(0x30 + number) for number in range(10)}
)
WINDOWS_KEY_CODES.update(
    {chr(code).lower(): key_codes(code) for code in range(0x41, 0x5B)}
)
WINDOWS_KEY_CODES.update(
    {f"numpad_{number}": key_codes(0x60 + number) for number in range(10)}
)
WINDOWS_KEY_CODES.update(
    {f"f{number}": key_codes(0x6F + number) for number in range(1, 25)}
)

ALIASES = {
    "esc": "escape",
    "return": "enter",
    "ctrl": "control",
    "left_ctrl": "left_control",
    "right_ctrl": "right_control",
    "pageup": "page_up",
    "pagedown": "page_down",
    "spacebar": "space",
}

# Type for the return struct of PsychHID('KbQueueGetEvent')
PsychHIDEvent = TypedDict(
    "PsychHIDEvent",
    {
        "Type": float,
        "Time": float,
        "Pressed": float,
        "Keycode": float,
        "CookedKey": float,
        "ButtonStates": float,
        "Motion": float,
        "X": float,
        "Y": float,
        "NormX": float,
        "NormY": float,
        "Valuators": npt.NDArray[np.float64],
    },
)


def normalize_key_name(key: str) -> str:
    name = key.strip().lower().replace("-", "_").replace(" ", "_")
    name = ALIASES.get(name, name)
    if name not in WINDOWS_KEY_CODES:
        available = ", ".join(sorted(WINDOWS_KEY_CODES))
        raise ValueError(f"Unknown keyboard key {key!r}. Available keys: {available}")
    return name


def resolve_key(key: str) -> tuple[int, ...]:
    return WINDOWS_KEY_CODES[normalize_key_name(key)]


@dataclass(slots=True)
class KeyEvent:
    key: str  # Normalized virtual key name
    pressed: bool  # True for press events, False for release events
    timestamp: float  # Time of the event in seconds
    raw_code: int  # DirectInput scan code (one-based)
    sample: int | None = None  # Sample number at the time of the event (set by engine)


class KeyboardQueue:

    def __init__(self, keys: set[str]) -> None:
        """
        Initialize the keyboard queue with the requested keys.

        The constructor creates an internal reverse lookup table mapping
        PsychHID key codes to logical key names.

        Example:
            >>> keys = {"space", "escape", "a", "shift"}
            >>> keyboard = KeyboardQueue(keys)
            >>> keyboard._code_names
            {27: 'escape', 32: 'space', 65: 'a', 160: 'shift', 161: 'shift'}
        """
        # Reverse lookup to convert PsychHID integer codes back to logical names.
        self._code_names: dict[int, str] = {
            code: key for key in sorted(keys) for code in resolve_key(key)
        }

        self._started: bool = False
        self._stopped: bool = False
        self._error: BaseException | None = None
        self._queue: SimpleQueue[KeyEvent] = SimpleQueue()
        self._thread: Thread = Thread(target=self._worker, daemon=True)

    @property
    def pending_events(self) -> Iterator[KeyEvent]:
        """
        Allows iterating over pending events, draining the queue.
            >>> for event in keyboard.pending_events:
            ...     print(event)

        Also allows draining the queue by iterating over this property.
            >>> for _ in self.pending_events:
            ...     pass
        """
        while True:
            try:
                yield self._queue.get_nowait()
            except Empty:
                return

    @property
    def error(self) -> BaseException | None:
        """Exception raised by the worker thread, if it has failed."""
        return self._error

    def is_started(self) -> bool:
        return self._started

    def is_stopped(self) -> bool:
        return self._stopped

    def start(self) -> None:
        """
        Start the PsychHID keyboard queue.
        Starts a worker thread to bridge PsychHID -> SimpleQueue.
        """
        if self.is_started():
            raise RuntimeError("Keyboard queue already started")
        if self.is_stopped():
            raise RuntimeError("KeyboardQueue cannot be restarted once stopped")

        # Converts requested key to PsychHID key mask
        mask = [0] * 256
        for code in self._code_names:
            mask[code - 1] = 1  # PsychHID one-based keycode -> zero-based mask index

        PsychHID("KbQueueCreate", None, mask, 0, 10_000, 0, None)
        try:
            PsychHID("KbQueueStart", None)
            GetSecs()  # Warm up Psychtoolbox's high-resolution Windows clock.
            self._thread.start()
            self._started = True
        except Exception:
            PsychHID("KbQueueRelease", None)
            raise

    def stop(self) -> None:
        if not self.is_started():
            return

        self._stopped = True
        self._started = False
        self._thread.join()

        try:
            PsychHID("KbQueueStop", None)
        finally:
            PsychHID("KbQueueRelease", None)
            for _ in self.pending_events:
                pass

    def _worker(self) -> None:
        """
        Bridges PsychHID into a queue the audio callback can drain:
        PsychHID -> SimpleQueue -> AudioCallback (callback never calls PsychHID directly).
        """
        try:
            while not self.is_stopped():
                hid_event = self._hid_get_event(0.0)

                # If a HID event is available, convert it to KeyEvent and put it on the queue
                while hid_event is not None:
                    key_event = self._to_key_event(hid_event)
                    if key_event is not None:
                        self._queue.put(key_event)

                    # Drain any other already-queued HID events without blocking
                    hid_event = self._hid_get_event(0.0)

                # Release the GIL so the audio callback can run.
                sleep(0.001)
        except BaseException as error:
            self._error = error
            self._stopped = True

    def _to_key_event(self, raw: PsychHIDEvent) -> KeyEvent | None:
        is_keypress = int(raw["Type"]) == 0
        if not is_keypress:
            return None

        raw_code = int(raw["Keycode"])
        key = self._code_names.get(raw_code)
        if key is None:
            return None

        return KeyEvent(
            key=key,
            pressed=bool(raw["Pressed"]),
            timestamp=raw["Time"],
            raw_code=raw_code,
        )

    def _hid_get_event(self, max_wait_time: float = 0.0) -> PsychHIDEvent | None:
        """KbQueueGetEvent: waits only when the HID queue is empty and max_wait_time > 0."""
        raw, _ = PsychHID("KbQueueGetEvent", None, max_wait_time)
        return raw if isinstance(raw, dict) else None


if __name__ == "__main__":
    import time

    keys = {"space", "escape", "a", "shift"}
    keyboard = KeyboardQueue(keys)
    keyboard.start()

    while True:
        for event in keyboard.pending_events:
            print(event)
        time.sleep(0.01)
