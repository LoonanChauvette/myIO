from __future__ import annotations

from dataclasses import dataclass
from queue import SimpleQueue
from threading import Event, Thread
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
    pressed: bool  # True if the key is pressed, False otherwise
    timestamp: float  # Time of the event in seconds
    raw_code: int  # DirectInput scan codes (one-based)
    sample: int | None = None  # Sample number at the time of the event (set by engine)


class KeyboardQueue:
    """Small lifecycle wrapper around the PsychHID keyboard queue."""

    def __init__(self, keys: set[str], event_queue: SimpleQueue[KeyEvent]) -> None:
        """
        Initialize the keyboard queue with the requested keys.

        The constructor creates an internal reverse lookup table mapping
        PsychHID key codes to logical key names.

        Example:
            >>> keys = {"space", "escape", "a", "shift"}
            >>> keyboard = _KeyboardQueue(keys)
            >>> keyboard._code_names
            {27: 'escape', 32: 'space', 65: 'a', 160: 'shift', 161: 'shift'}
        """
        self._code_names: dict[int, str] = {}
        self._started: bool = False

        self._event_queue: SimpleQueue[KeyEvent] = event_queue
        self._thread: Thread | None = None
        self._stop: Event = Event()
        self._error: BaseException | None = None

        # Reverse lookup used to convert PsychHID integer codes back to logical names.
        for key in sorted(keys):
            for code in resolve_key(key):
                self._code_names[code] = key

    def start(self) -> None:
        """
        Start the PsychHID keyboard queue.

        The requested key names are converted into a PsychHID key mask.
        PsychHID uses one-based keycode indexing internally,
        Python uses zero-based indexing, so mask keycodes are shifted by one.
        """
        if self._started:
            raise RuntimeError("Keyboard queue already started")

        mask = [0] * 256
        for code in self._code_names:
            mask[code - 1] = 1

        try:
            self._create_queue(mask)
            self._start_queue()
            GetSecs()  # Warm up Psychtoolbox's high-resolution Windows clock.
            self._thread = Thread(
                target=self._worker, name="myio-keyboard", daemon=True
            )
            self._thread.start()
            self._started = True
        except Exception:
            self._release_queue()
            raise

    def stop(self) -> None:
        if not self._started:
            return

        self._stop.set()
        if self._thread is not None:
            self._thread.join()

        try:
            self._stop_queue()
        finally:
            self._release_queue()
            self._started = False

    def poll(self) -> list[KeyEvent]:
        events: list[KeyEvent] = []

        for raw in self._iter_queue():
            event = self._to_key_event(raw)
            if event is not None:
                events.append(event)

        return events

    def _iter_queue(self) -> Iterator[PsychHIDEvent]:
        while True:
            event = self._get_event_from_queue()
            if event is None:
                break

            yield event

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

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                for event in self.poll():
                    self._event_queue.put(event)
                self._stop.wait(0.001)
        except BaseException as error:
            self._error = error
            self._stop.set()

    def _create_queue(self, mask: list[int]) -> None:
        """Wrapper around PsychHID('KbQueueCreate'): psychtoolbox.org/docs/PsychHID-KbQueueCreate"""
        PsychHID("KbQueueCreate", None, mask, 0, 10_000, 0, None)

    def _start_queue(self) -> None:
        """Wrapper for PsychHID('KbQueueStart'): psychtoolbox.org/docs/PsychHID-KbQueueStart"""
        PsychHID("KbQueueStart", None)

    def _release_queue(self) -> None:
        """Wrapper for PsychHID('KbQueueRelease'): psychtoolbox.org/docs/PsychHID-KbQueueRelease"""
        PsychHID("KbQueueRelease", None)

    def _stop_queue(self) -> None:
        """Wrapper for PsychHID('KbQueueStop'): psychtoolbox.org/docs/PsychHID-KbQueueStop"""
        PsychHID("KbQueueStop", None)

    def _get_available_events(self) -> int:
        """Wrapper for PsychHID('KbQueueFlush'): http://psychtoolbox.org/docs/PsychHID-KbQueueFlush"""
        n_avail = PsychHID("KbQueueFlush", None, 0)
        return int(n_avail)

    def _get_event_from_queue(self, max_wait_time: float = 0.0) -> PsychHIDEvent | None:
        """Wrapper for PsychHID('KbQueueGetEvent'): psychtoolbox.org/docs/PsychHID-KbQueueGetEvent"""
        raw, _ = PsychHID("KbQueueGetEvent", None, max_wait_time)
        return raw if isinstance(raw, dict) else None

    def _clear_queue(self) -> None:
        """Wrapper for PsychHID('KbQueueFlush'): http://psychtoolbox.org/docs/PsychHID-KbQueueFlush"""
        PsychHID("KbQueueFlush", None, 3)


if __name__ == "__main__":
    import time

    keys = {"space", "escape", "a", "shift"}
    queue = SimpleQueue()
    keyboard = KeyboardQueue(keys, queue)
    keyboard.start()

    while True:
        try:
            raw = queue.get_nowait()
            if raw is not None:
                print(raw)
        except Exception:
            pass

        time.sleep(0.01)
