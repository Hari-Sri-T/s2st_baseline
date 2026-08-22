"""
Entry point for Week 1, Days 1-2: get speech -> translation -> speech running
end to end and save input/output audio for 2-3 Indic languages.

USAGE
-----
1. Drop source recordings into test_audio/ using this naming convention:

       {speaker}_{source_lang}_{tag}.wav

   e.g.  speakerA_hi_neutral.wav
         speakerA_te_excited.wav
         speakerB_mr_neutral.wav

   source_lang must be one of the keys in config.LANGUAGES (hi/te/mr/en).
   tag is free text (neutral/happy/sad/etc.) - not parsed yet, just for
   your own bookkeeping ahead of Week 2's emotion testing.

2. Set TARGET_LANGS below to whichever languages you want each file
   translated into.

3. Run:  python run_baseline.py

Outputs land in results/: <run_id>__output.wav, <run_id>__ref.wav, and a
running results/metadata.json log (transcript, translation, everything)
that Week 2's failure-matrix work will read from directly.
"""
import glob
import os

# MUST be set before torch/transformers get imported anywhere below.
# AI4Bharat's IndicF5 model.py wraps its vocoder in torch.compile(), which
# hits a device-tracing bug (dynamo confuses cpu/meta tensors) on newer
# PyTorch. Disabling dynamo makes torch.compile a harmless no-op - same
# output, just without the compile speedup.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from pipeline.baseline_pipeline import BaselinePipeline
import config

# Which languages to translate each source recording INTO.
# Keep this small at first - per roadmap doc, start with 2-3 languages.
TARGET_LANGS = ["hi", "te", "mr"]


def parse_filename(path: str):
    """speakerA_hi_neutral.wav -> ("speakerA", "hi", "neutral")"""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Filename '{path}' doesn't match {{speaker}}_{{lang}}_{{tag}}.wav convention"
        )
    speaker = parts[0]
    lang = parts[1]
    tag = "_".join(parts[2:]) if len(parts) > 2 else "untagged"
    return speaker, lang, tag


def main():
    audio_files = sorted(glob.glob(os.path.join(config.TEST_AUDIO_DIR, "*.wav")))
    if not audio_files:
        print(
            f"No .wav files found in {config.TEST_AUDIO_DIR}/. "
            "Add source recordings first (see docstring at top of this file)."
        )
        return

    print(f"Found {len(audio_files)} source file(s). Loading models (this takes a minute)...")
    pipeline = BaselinePipeline()

    total_runs = 0
    for audio_path in audio_files:
        speaker, src_lang, tag = parse_filename(audio_path)

        if src_lang not in config.LANGUAGES:
            print(f"[skip] {audio_path}: unknown source_lang '{src_lang}'")
            continue

        for tgt_lang in TARGET_LANGS:
            if tgt_lang == src_lang:
                continue  # no point translating a language into itself

            run_id = f"{speaker}_{src_lang}_{tag}__to_{tgt_lang}"
            print(f"\n=== {run_id} ===")
            try:
                pipeline.run(
                    source_audio_path=audio_path,
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    run_id=run_id,
                )
                total_runs += 1
            except Exception as e:
                print(f"[ERROR] {run_id} failed: {e}")

    print(f"\nDone. {total_runs} runs completed. See {config.RESULTS_DIR}/metadata.json")


if __name__ == "__main__":
    main()
