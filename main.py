import sounddevice as sd

from dataclasses import dataclass

from enum import Enum


class Kind(Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class Device:
    name: str
    index: int
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_low_input_latency: float
    default_low_output_latency: float
    default_high_input_latency: float
    default_high_output_latency: float
    default_sample_rate: float

def main():

    devices: sd.DeviceList = sd.query_devices()


if __name__ == "__main__":
    main()
