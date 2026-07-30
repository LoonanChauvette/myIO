from typing import Protocol


class CallbackTime(Protocol):
    currentTime: float
    inputBufferAdcTime: float
    outputBufferDacTime: float


class Clock:
    def __init__(self, samplerate: float, blocksize: int) -> None:
        self.samplerate: float = samplerate
        self.blocksize: int = blocksize

        self.frame: int = 0
        self.next_frame: int = 0
        self.current_time: float = 0.0
        self.output_time: float = 0.0

    def tick(self, frames: int, time: CallbackTime) -> None:
        self.frame = self.next_frame
        self.next_frame += frames
        self.current_time = time.currentTime
        self.output_time = time.outputBufferDacTime
