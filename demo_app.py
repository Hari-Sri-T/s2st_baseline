import os
import tempfile
import gradio as gr
from pipeline.baseline_pipeline import BaselinePipeline
import config
import time

# Initialize pipeline lazily to save resources if just checking UI layout
pipeline = None

def init_pipeline():
    global pipeline
    if pipeline is None:
        print("Initializing BaselinePipeline for Demo...")
        pipeline = BaselinePipeline()
    return pipeline

def save_profile(audio_path, transcript_text):
    if not audio_path:
        return None, "Please record your audio first."
    if not transcript_text:
        return None, "Transcript text is required."
        
    # We will just pass the paths directly during the call.
    return audio_path, f"Profile Saved Successfully! You can now use the Phone Call Simulator."

def simulate_call(profile_audio, profile_text, call_audio, source_lang, target_lang):
    if not call_audio:
        return "Please record a message first.", None, None, None
        
    p = init_pipeline()
    
    # Generate a unique run ID
    run_id = f"demo_call_{int(time.time())}"
    
    # Convert language names to language codes
    src_code = None
    tgt_code = None
    for code, info in config.LANGUAGES.items():
        if info["name"] == source_lang:
            src_code = code
        if info["name"] == target_lang:
            tgt_code = code
            
    if not src_code or not tgt_code:
        return "Invalid language selection.", None, None, None
        
    try:
        # Run the pipeline
        result = p.run(
            source_audio_path=call_audio,
            source_lang=src_code,
            target_lang=tgt_code,
            run_id=run_id,
            custom_ref_audio=profile_audio,
            custom_ref_text=profile_text
        )
        
        latency = result.get("latency", {})
        latency_str = (
            f"ASR: {latency.get('asr_seconds', 0)}s | "
            f"Translation: {latency.get('mt_seconds', 0)}s | "
            f"TTS: {latency.get('tts_seconds', 0)}s | "
            f"Total: {latency.get('total_seconds', 0)}s"
        )
        
        return (
            result["transcript"], 
            result["translated_text"], 
            result["output_audio"], 
            latency_str
        )
    except Exception as e:
        return f"Error occurred: {str(e)}", None, None, None

def build_app():
    # Setup languages list
    lang_choices = [info["name"] for info in config.LANGUAGES.values()]
    
    with gr.Blocks(title="S2ST Real-time Phone Call Demo") as app:
        gr.Markdown("# 📞 Speech-to-Speech Translation Demo")
        gr.Markdown("This demo simulates a real-time phone call translating your voice into another language while preserving your voice, emotion, and prosody.")
        
        # We need a state variable to hold the saved profile audio path and text
        profile_audio_state = gr.State(None)
        profile_text_state = gr.State("Hello, my name is testing user, and I am recording my voice profile for this application.")
        
        with gr.Tabs():
            with gr.Tab("1. Voice Profile Setup"):
                gr.Markdown("### Create your Voice Profile")
                gr.Markdown("Read the transcript below clearly so the AI can clone your voice properties (timbre, emotion, pitch).")
                
                transcript_box = gr.Textbox(
                    value="Hello, my name is testing user, and I am recording my voice profile for this application.",
                    label="Read this transcript aloud:",
                    lines=2
                )
                
                record_profile = gr.Audio(sources=["microphone"], type="filepath", label="Record your Voice Profile")
                
                save_btn = gr.Button("Save Profile", variant="primary")
                profile_status = gr.Textbox(label="Status", interactive=False)
                
                save_btn.click(
                    fn=save_profile,
                    inputs=[record_profile, transcript_box],
                    outputs=[profile_audio_state, profile_status]
                ).then(
                    fn=lambda x: x,
                    inputs=transcript_box,
                    outputs=profile_text_state
                )

            with gr.Tab("2. Phone Call Simulator"):
                gr.Markdown("### Simulate a Call")
                gr.Markdown("Make sure you have saved your profile in the first tab. Then, record a message in your source language.")
                
                with gr.Row():
                    src_lang_dropdown = gr.Dropdown(choices=lang_choices, value="English", label="Your Language (Source)")
                    tgt_lang_dropdown = gr.Dropdown(choices=lang_choices, value="Hindi", label="Other Person's Language (Target)")
                
                call_recording = gr.Audio(sources=["microphone"], type="filepath", label="Speak into the phone")
                
                translate_btn = gr.Button("Translate & Send", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        asr_output = gr.Textbox(label="What you said (ASR Transcript)")
                        translation_output = gr.Textbox(label="Translated Text")
                    with gr.Column():
                        audio_output = gr.Audio(label="What the other person hears (Target Audio)")
                        latency_output = gr.Textbox(label="Latency Breakdown")
                
                translate_btn.click(
                    fn=simulate_call,
                    inputs=[profile_audio_state, profile_text_state, call_recording, src_lang_dropdown, tgt_lang_dropdown],
                    outputs=[asr_output, translation_output, audio_output, latency_output]
                )

    return app

if __name__ == "__main__":
    app = build_app()
    app.launch(share=True, debug=True)
