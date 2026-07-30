from collections.abc import Iterator
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue


@dataclass(slots=True)
class BaseEvent:
    timestamp: float
    sample: int | None = field(
        default=None, kw_only=True
    )  # Sample number at the time of the event (set by engine)


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


@dataclass(slots=True)
class KeyEvent(BaseEvent):
    key: str
    pressed: bool
    raw_code: int
