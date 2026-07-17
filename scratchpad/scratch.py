from myio import AudioEngine, select_audio_config

if __name__ == "__main__":
    # engine = AudioEngine.from_args(samplerate=48000, channels=2, latency="high")
    config = select_audio_config()
    engine = AudioEngine.from_dict(config)
    engine.start()
