"""
Picks the TTS backend based on config.TTS_BACKEND so pipeline.py doesn't
need to know which one is active. Lets you A/B IndicF5 vs F5-TTS by
flipping one line in config.py (roadmap doc explicitly calls for evaluating
both under Stage 1).
"""
def build_tts(backend: str, checkpoint: str, device: str):
    if backend == "indicf5":
        from tts.indicf5_tts import IndicF5TTS
        return IndicF5TTS(checkpoint=checkpoint, device=device)
    elif backend == "f5":
        from tts.f5_tts import F5TTSWrapper
        return F5TTSWrapper(checkpoint=checkpoint, device=device)
    else:
        raise ValueError(f"Unknown TTS_BACKEND '{backend}'. Use 'indicf5' or 'f5'.")
