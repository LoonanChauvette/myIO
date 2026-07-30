from collections.abc import Iterator
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import Self, TypedDict

import numpy as np
import numpy.typing as npt
import sounddevice as sd


@dataclass(slots=True)
class BaseEvent:
    timestamp: float
    # Output sample at the time of the event (set by engine)
    sample: int | None = field(default=None, kw_only=True)


class EventQueue(SimpleQueue):
    @property
    def pending_events(self) -> Iterator[BaseEvent]:
        """
        Allows iterating over pending events, draining the queue.
            >>> for event in queue.pending_events:
            ...     print(event)

        Also allows draining the queue by iterating over this property.
            >>> for _ in queue.pending_events:
            ...     pass
        """
        while True:
            try:
                yield self.get_nowait()
            except Empty:
                return


@dataclass(slots=True)
class CallbackEvent(BaseEvent):
    output_underflow: bool = False
    output_overflow: bool = False
    input_underflow: bool = False
    input_overflow: bool = False

    @classmethod
    def from_flags(cls, flags: sd.CallbackFlags, timestamp: float, sample: int) -> Self:
        return cls(
            timestamp=timestamp,
            sample=sample,
            output_underflow=flags.output_underflow,
            output_overflow=flags.output_overflow,
            input_underflow=flags.input_underflow,
            input_overflow=flags.input_overflow,
        )


@dataclass(slots=True)
class KeyEvent(BaseEvent):
    key: str
    pressed: bool
    raw_code: int

    @classmethod
    def from_hid_event(cls, raw: PsychHIDEvent, codes: dict[int, str]) -> Self | None:
        if int(raw["Type"]) != 0:  # If Type is not 0, not a keypress
            return None

        key = codes.get(int(raw["Keycode"]))
        if key is None:
            return None

        return cls(
            key=key,
            pressed=bool(raw["Pressed"]),
            timestamp=raw["Time"],
            raw_code=int(raw["Keycode"]),
        )


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
