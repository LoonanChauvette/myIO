from myio import AudioEngine

if __name__ == "__main__":
    engine = AudioEngine.from_args(samplerate=48000, channels=2, latency="high")
    engine.start()
