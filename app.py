from __future__ import annotations
import gradio as gr
import warnings
import os
import traceback
import logging
from datetime import datetime
from typing import Tuple
from elevenlabs import ElevenLabs
warnings.filterwarnings("ignore")

# ==================== LOGGING SETUP ====================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"asr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("ASR System Starting")
logger.info(f"Log file: {log_file}")
logger.info("="*60)

# ==================== CONFIGURATION ====================
DESCRIPTION = "ASR with Speaker Diarization using Eleven Labs"

# Mapping from user-friendly language names to Eleven Labs 3-letter language codes
LANGUAGE_NAME_TO_CODE = {
    "English": "eng", "Spanish": "spa", "French": "fra", "German": "deu",
    "Italian": "ita", "Portuguese": "por", "Dutch": "nld", "Hindi": "hin",
    "Japanese": "jpn", "Chinese": "zho", "Finnish": "fin", "Korean": "kor",
    "Polish": "pol", "Russian": "rus", "Turkish": "tur", "Ukrainian": "ukr",
    "Swedish": "swe", "Danish": "dan", "Norwegian": "nor", "Romanian": "ron"
}

# ==================== UTILITY FUNCTIONS ====================
def safe_function(func):
    """Decorator to safely execute functions with error handling and logging."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return f"An error occurred: {str(e)}", "", "Error"
    return wrapper

# ==================== CORE TRANSCRIPTION LOGIC ====================
progress_callback_ref = None

def format_diarized_transcript(response) -> Tuple[str, str]:
    """Formats the diarized response from Eleven Labs into a readable transcript."""
    segments = []
    current_speaker = None
    current_segment = []
    speaker_words = {}

    for word in response.words:
        speaker = word.get('speaker_id', 'unknown_speaker')
        if speaker not in speaker_words:
            speaker_words[speaker] = []
        speaker_words[speaker].append(word['text'])

        if speaker != current_speaker:
            if current_segment:
                speaker_name = f"Speaker {int(current_speaker.split('_')[-1]) + 1}"
                segments.append(f"**{speaker_name}:** {' '.join(current_segment)}")
            current_segment = [word['text']]
            current_speaker = speaker
        else:
            current_segment.append(word['text'])
    
    # Append the last segment
    if current_segment:
        speaker_name = f"Speaker {int(current_speaker.split('_')[-1]) + 1}"
        segments.append(f"**{speaker_name}:** {' '.join(current_segment)}")

    full_transcription = "\n\n".join(segments)

    # Calculate speaker statistics
    diarization_info = f"Speakers Found: {len(speaker_words)}\n"
    total_words = len(response.words)
    for speaker, words in sorted(speaker_words.items()):
        speaker_name = f"Speaker {int(speaker.split('_')[-1]) + 1}"
        word_count = len(words)
        percentage = (word_count / total_words) * 100 if total_words > 0 else 0
        diarization_info += f"- {speaker_name}: {word_count} words ({percentage:.1f}%)\n"
        
    return full_transcription, diarization_info

@safe_function
def transcribe_with_elevenlabs(
    api_key: str,
    audio_path: str,
    language_name: str,
    enable_diarization: bool,
    progress_callback=None
) -> Tuple[str, str, str]:
    """
    Transcribes audio using the Eleven Labs API, with an option for diarization.
    """
    global progress_callback_ref
    progress_callback_ref = progress_callback

    if not api_key:
        return "Error: Eleven Labs API key is missing.", "", "Failed"
    if not audio_path:
        return "Please upload an audio file.", "", "Failed"

    lang_code = LANGUAGE_NAME_TO_CODE.get(language_name, "eng")
    logger.info(f"Starting transcription for '{language_name}' ({lang_code}). Diarization: {enable_diarization}")

    try:
        if progress_callback:
            progress_callback("Initializing Eleven Labs client...")
        client = ElevenLabs(api_key=api_key)

        if progress_callback:
            progress_callback("Uploading and transcribing audio...")
        
        with open(audio_path, "rb") as f:
            response = client.speech_to_text.convert(
                file=f,
                model="scribe_v1",
                language=lang_code,
                diarize=enable_diarization
            )

        logger.info("Successfully received response from Eleven Labs API.")
        
        full_transcription = response.text
        diarization_info = "Diarization disabled."

        if enable_diarization and hasattr(response, 'words') and any('speaker_id' in w for w in response.words):
            if progress_callback:
                progress_callback("Formatting diarized transcript...")
            full_transcription, diarization_info = format_diarized_transcript(response)
        
        status = "✅ Transcription successful."
        logger.info(status)
        return full_transcription, diarization_info, status

    except Exception as e:
        logger.error(f"Eleven Labs API error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error during transcription: {str(e)}", "", "Failed"


# ==================== GRADIO INTERFACE ====================
def transcribe_wrapper(api_key, audio, language, enable_diarization, progress=gr.Progress()):
    """Wrapper for Gradio to handle the transcription process and progress updates."""
    progress(0, desc="Starting...")

    def update_progress(msg):
        progress(0.5, desc=msg)

    result = transcribe_with_elevenlabs(
        api_key, audio, language, enable_diarization, update_progress
    )

    progress(1.0, desc="Complete!")
    return result

def get_recent_logs():
    """Fetches the last 20 lines from the log file to display in the UI."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-20:])
    except Exception as e:
        logger.error(f"Could not read log file: {e}")
        return "Log file not accessible."

# ==================== UI LAYOUT ====================
with gr.Blocks(title="ASR with Eleven Labs", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🎙️ ASR with Diarization (Eleven Labs Edition)
    
    This application uses the Eleven Labs API to perform speech-to-text with speaker diarization.
    Enter your API key and an audio file to begin.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            api_key_input = gr.Textbox(
                label="Eleven Labs API Key",
                type="password",
                placeholder="Enter your API key here..."
            )
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            
            language_dropdown = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGE_NAME_TO_CODE.keys()),
                value="English"
            )
            
            diarization_checkbox = gr.Checkbox(label="Enable Speaker Diarization", value=True)
            
            process_btn = gr.Button("🚀 Process Audio", variant="primary")
            
        with gr.Column(scale=3):
            status_output = gr.Textbox(label="Processing Status", lines=1, interactive=False)
            diarization_output = gr.Textbox(label="Speaker Info", lines=4, interactive=False)
            transcription_output = gr.Textbox(label="Transcript", lines=15, show_copy_button=True, interactive=False)

    with gr.Accordion("📋 System Logs", open=False):
        log_display = gr.Textbox(
            label="Recent Logs",
            lines=10,
            interactive=False,
            value=get_recent_logs
        )
        refresh_logs_btn = gr.Button("🔄 Refresh Logs")

    # ==================== EVENT HANDLERS ====================
    process_btn.click(
        fn=transcribe_wrapper,
        inputs=[api_key_input, audio_input, language_dropdown, diarization_checkbox],
        outputs=[transcription_output, diarization_output, status_output]
    )
    
    refresh_logs_btn.click(
        fn=get_recent_logs,
        outputs=log_display
    )

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.queue().launch(share=True, debug=True)
