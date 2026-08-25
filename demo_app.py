"""
Gradio Demo App for S2ST — Expressive Indic Speech-to-Speech Translation.

Tab 1: Voice Profile Setup — record a clean reference clip for voice cloning.
Tab 2: Phone Call Simulator — speak in one language, hear it in another in your voice.

All models are loaded eagerly at startup so inference calls have zero cold-start delay.
"""
import os
import time

import gradio as gr

from pipeline.baseline_pipeline import BaselinePipeline
import config


# ---------------------------------------------------------------------------
# Model warm-up — runs once at startup, not on the first request
# ---------------------------------------------------------------------------
print("=" * 60)
print("[STARTUP] Loading all models into GPU memory. Please wait...")
print("=" * 60)

pipeline = BaselinePipeline()

# Pre-warm ALL three IndicTrans2 direction models so no request ever waits
# for a 4.8 GB model download mid-call.
pipeline.translator.warmup_all()

print("[STARTUP] All models loaded! Demo is ready.")
print("=" * 60)


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------

def save_profile(audio_path, transcript_text):
    if not audio_path:
        return None, None, "❌ Please record your audio first."
    if not transcript_text or not transcript_text.strip():
        return None, None, "❌ Transcript text is required."
    return audio_path, transcript_text, "✅ Voice profile saved! Go to Tab 2 to make a call."


def simulate_call(profile_audio, profile_text, call_audio, source_lang, target_lang):
    if not call_audio:
        return "❌ Please record a message first.", None, None, None

    if not profile_audio:
        return (
            "⚠️ No voice profile set. Go to Tab 1 to record your voice profile first.",
            None, None, None
        )

    if source_lang == target_lang:
        return "❌ Source and target languages must be different.", None, None, None

    # Map display names back to lang codes
    src_code = next((c for c, info in config.LANGUAGES.items() if info["name"] == source_lang), None)
    tgt_code = next((c for c, info in config.LANGUAGES.items() if info["name"] == target_lang), None)

    if not src_code or not tgt_code:
        return "❌ Invalid language selection.", None, None, None

    run_id = f"demo_call_{int(time.time())}"

    try:
        result = pipeline.run(
            source_audio_path=call_audio,
            source_lang=src_code,
            target_lang=tgt_code,
            run_id=run_id,
            custom_ref_audio=profile_audio,
            custom_ref_text=profile_text,
        )

        latency = result.get("latency", {})
        latency_str = (
            f"🎙️  ASR:         {latency.get('asr_seconds', 0):.2f}s\n"
            f"🌐  Translation: {latency.get('mt_seconds', 0):.2f}s\n"
            f"🔊  TTS:         {latency.get('tts_seconds', 0):.2f}s\n"
            f"⏱️  Total:       {latency.get('total_seconds', 0):.2f}s"
        )

        return (
            result.get("transcript", ""),
            result.get("translated_text", ""),
            result.get("output_audio"),
            latency_str,
        )

    except Exception as e:
        import traceback
        return f"❌ Error: {e}\n\n{traceback.format_exc()}", None, None, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

LANG_CHOICES = [info["name"] for info in config.LANGUAGES.values()]

PROFILE_TRANSCRIPT = (
    "Hello! I am recording my voice profile for this speech translation system. "
    "My voice will be cloned to speak in another language."
)

with gr.Blocks(title="S2ST Demo — Indic Speech Translation") as app:

    # ---------- shared state ----------
    profile_audio_state = gr.State(None)
    profile_text_state  = gr.State(None)

    # ---------- header ----------
    gr.Markdown(
        """
        # 📞 Real-time Indic Speech-to-Speech Translation
        > **Cascaded pipeline:** Whisper ASR → IndicTrans2 → IndicF5 zero-shot voice cloning

        This demo simulates a cross-language phone call.
        Your voice is cloned so the translated speech sounds like *you*.
        """
    )

    with gr.Tabs():

        # ══════════════════════════════════════════════════════
        # TAB 1 — Voice Profile
        # ══════════════════════════════════════════════════════
        with gr.Tab("1. 🎤 Voice Profile Setup"):
            gr.Markdown(
                "### Step 1: Create your Voice Profile\n"
                "Read the sentence below **clearly and naturally** into the microphone. "
                "This recording is used to clone your voice into the target language."
            )

            transcript_box = gr.Textbox(
                value=PROFILE_TRANSCRIPT,
                label="📄 Read this sentence aloud:",
                lines=3,
            )

            record_profile = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️ Record your voice profile here",
            )

            save_btn = gr.Button("💾 Save Profile", variant="primary", size="lg")
            profile_status = gr.Textbox(label="Status", interactive=False, lines=1)

            save_btn.click(
                fn=save_profile,
                inputs=[record_profile, transcript_box],
                outputs=[profile_audio_state, profile_text_state, profile_status],
            )

        # ══════════════════════════════════════════════════════
        # TAB 2 — Phone Call Simulator
        # ══════════════════════════════════════════════════════
        with gr.Tab("2. 📱 Phone Call Simulator"):
            gr.Markdown(
                "### Step 2: Make a Call\n"
                "Select your language and the **other person's language**, "
                "then speak into the microphone."
            )

            with gr.Row():
                src_lang_dd = gr.Dropdown(
                    choices=LANG_CHOICES, value="English",
                    label="🗣️ Your language (source)"
                )
                tgt_lang_dd = gr.Dropdown(
                    choices=LANG_CHOICES, value="Telugu",
                    label="👂 Other person's language (target)"
                )

            call_recording = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="📞 Speak your message",
            )

            translate_btn = gr.Button("🚀 Translate & Send", variant="primary", size="lg")

            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column():
                    asr_output         = gr.Textbox(label="📝 What you said (ASR transcript)", lines=3)
                    translation_output = gr.Textbox(label="🌐 Translated text", lines=3)
                with gr.Column():
                    audio_output   = gr.Audio(label="🔊 What the other person hears (your voice)")
                    latency_output = gr.Textbox(label="⏱️ Latency Breakdown", lines=5)

            translate_btn.click(
                fn=simulate_call,
                inputs=[
                    profile_audio_state, profile_text_state,
                    call_recording, src_lang_dd, tgt_lang_dd,
                ],
                outputs=[asr_output, translation_output, audio_output, latency_output],
            )


if __name__ == "__main__":
    app.launch(share=True, debug=False)
