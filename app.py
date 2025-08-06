from __future__ import annotations
import torch
import torchaudio
import gradio as gr
import gc
import warnings
import numpy as np
from typing import List, Tuple, Dict
import time
import os
import shutil
import traceback
import logging
from datetime import datetime
warnings.filterwarnings("ignore")

# ==================== LOGGING SETUP ====================
# Create comprehensive logging
log_dir = "/content/logs" if os.path.exists("/content") else "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"asr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Configure logging
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

# ==================== ENVIRONMENT SETUP ====================
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/content/cache' if os.path.exists("/content") else './cache'
os.environ['HF_HOME'] = os.environ['TRANSFORMERS_CACHE']

# Check device with error handling
try:
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Found: {gpu_properties.name}")
        logger.info(f"GPU Memory: {gpu_properties.total_memory / 1024**3:.1f} GB")
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        logger.warning("No GPU available, using CPU")
        device = "cpu"
except Exception as e:
    logger.error(f"Error checking GPU: {e}")
    device = "cpu"

DESCRIPTION = "IndicConformer ASR with Speaker Diarization & Translation"

LANGUAGE_NAME_TO_CODE = {
    "Assamese": "as", "Bengali": "bn", "Bodo": "br", "Dogri": "doi",
    "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Kashmiri": "ks",
    "Konkani": "kok", "Maithili": "mai", "Malayalam": "ml", "Manipuri": "mni",
    "Marathi": "mr", "Nepali": "ne", "Odia": "or", "Punjabi": "pa",
    "Sanskrit": "sa", "Santali": "sat", "Sindhi": "sd", "Tamil": "ta",
    "Telugu": "te", "Urdu": "ur"
}

# Global model variables
asr_model = None
asr_processor = None
translation_model = None
model_load_status = "Not started"

# ==================== UTILITY FUNCTIONS ====================
def safe_function(func):
    """Decorator to safely execute functions with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)[:100]}"
    return wrapper

@safe_function
def check_memory():
    """Safely check GPU memory status"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - allocated
            status = f"💾 GPU: {allocated:.1f}GB used | {free:.1f}GB free | {total:.1f}GB total"
            logger.debug(f"Memory check: {status}")
            return status
        else:
            return "💾 Running on CPU (No GPU available)"
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        return "💾 Memory status unavailable"

@safe_function
def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        path = "/content" if os.path.exists("/content") else "."
        stat = shutil.disk_usage(path)
        gb_free = stat.free / (1024**3)
        gb_total = stat.total / (1024**3)
        logger.info(f"Disk space: {gb_free:.1f}GB free / {gb_total:.1f}GB total")
        return gb_free
    except Exception as e:
        logger.error(f"Disk check failed: {e}")
        return 10  # Return default value

def clear_model_cache():
    """Clear HuggingFace cache to fix stuck downloads"""
    try:
        cache_dirs = [
            os.environ.get('TRANSFORMERS_CACHE', './cache'),
            '/content/cache',
            '~/.cache/huggingface',
            '/root/.cache/huggingface'
        ]
        
        for cache_dir in cache_dirs:
            cache_path = os.path.expanduser(cache_dir)
            if os.path.exists(cache_path):
                try:
                    logger.info(f"Clearing cache at {cache_path}")
                    shutil.rmtree(cache_path)
                    os.makedirs(cache_path, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not clear {cache_path}: {e}")
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")

# ==================== MODEL LOADING ====================
def load_asr_model_robust(max_retries=3):
    """Load ASR model with comprehensive error handling"""
    global asr_model, asr_processor, model_load_status
    
    logger.info("Starting ASR model loading")
    model_load_status = "Loading..."
    
    # Clean up existing model
    if asr_model is not None:
        logger.info("Cleaning up existing model")
        del asr_model
        asr_model = None
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Check disk space
    free_space = check_disk_space()
    if free_space < 5:
        logger.warning(f"Low disk space: {free_space:.1f}GB")
        clear_model_cache()
    
    model_name = "ai4bharat/indic-conformer-600m-multilingual"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading attempt {attempt + 1}/{max_retries}")
            
            from transformers import AutoModel, AutoConfig
            
            # Test connection
            logger.info("Testing HuggingFace connection...")
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=os.environ['TRANSFORMERS_CACHE']
            )
            logger.info("Connection successful")
            
            # Load model
            logger.info("Downloading model (2-3 minutes)...")
            asr_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                resume_download=True,
                local_files_only=False,
                force_download=False,
            )
            
            logger.info("Model downloaded, moving to device...")
            asr_model = asr_model.to(device)
            asr_model.eval()
            
            # Note about processor
            logger.info("Note: This model doesn't use a processor (expected behavior)")
            asr_processor = None
            
            model_load_status = "✅ Model loaded successfully!"
            logger.info("ASR Model loaded successfully!")
            torch.cuda.empty_cache()
            return True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            model_load_status = f"⚠️ Loading failed (attempt {attempt + 1})"
            
            if attempt == max_retries - 1:
                logger.error("All attempts failed, trying fallback model")
                return try_alternative_model()
            
            time.sleep(3)
            torch.cuda.empty_cache()
            gc.collect()
    
    model_load_status = "❌ Model loading failed"
    return False

def try_alternative_model():
    """Load a simpler fallback model"""
    global asr_model, asr_processor, model_load_status
    
    try:
        logger.info("Loading fallback wav2vec2 model...")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        model_name = "facebook/wav2vec2-large-xlsr-53-hindi"
        asr_processor = Wav2Vec2Processor.from_pretrained(model_name)
        asr_model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        model_load_status = "✅ Fallback model loaded"
        logger.info("Fallback model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Fallback model also failed: {e}")
        model_load_status = "❌ All models failed to load"
        return False

def load_translation_model_simple():
    """Load translation model"""
    global translation_model
    
    try:
        from googletrans import Translator
        translation_model = Translator()
        logger.info("Translation model loaded (Google Translate)")
        return True
    except Exception as e:
        logger.warning(f"Translation not available: {e}")
        return False

# ==================== AUDIO PROCESSING ====================
def simple_vad_diarization(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    min_speech_duration: float = 0.5
) -> List[Tuple[float, float, str]]:
    """Simple speaker diarization"""
    try:
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        
        energy = []
        for i in range(0, len(waveform) - frame_length, hop_length):
            frame = waveform[i:i + frame_length]
            energy.append(torch.sqrt(torch.mean(frame ** 2)).item())
        
        energy = np.array(energy)
        threshold = np.percentile(energy, 30)
        
        is_speech = energy > threshold
        
        segments = []
        start_idx = None
        current_speaker = "Speaker_1"
        
        for i in range(len(is_speech)):
            if is_speech[i] and start_idx is None:
                start_idx = i
            elif not is_speech[i] and start_idx is not None:
                start_time = start_idx * 0.010
                end_time = i * 0.010
                
                if end_time - start_time > min_speech_duration:
                    segments.append((start_time, end_time, current_speaker))
                    current_speaker = "Speaker_2" if current_speaker == "Speaker_1" else "Speaker_1"
                
                start_idx = None
        
        if start_idx is not None:
            segments.append((start_idx * 0.010, len(is_speech) * 0.010, current_speaker))
        
        return segments if segments else [(0, len(waveform)/sample_rate, "Speaker_1")]
        
    except Exception as e:
        logger.error(f"Diarization error: {e}")
        return [(0, len(waveform)/sample_rate, "Speaker_1")]

def translate_text_simple(text: str) -> str:
    """Translate text to English"""
    global translation_model
    
    if not text or not translation_model:
        return text
    
    try:
        result = translation_model.translate(text, dest='en')
        return result.text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

@safe_function
def process_audio_chunk(
    chunk: torch.Tensor,
    lang_code: str,
    chunk_idx: int,
    total_chunks: int
) -> str:
    """Process a single audio chunk"""
    global asr_model
    
    if asr_model is None:
        return "[Model not loaded]"
    
    try:
        # Ensure mono
        if chunk.dim() > 1 and chunk.shape[0] > 1:
            chunk = torch.mean(chunk, dim=0, keepdim=True)
        
        chunk = chunk.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                try:
                    # Direct model call (indic-conformer style)
                    transcription = asr_model(chunk, lang_code, "ctc")
                    
                    if isinstance(transcription, str):
                        return transcription.strip()
                    else:
                        return str(transcription).strip()
                        
                except Exception as e:
                    logger.debug(f"Direct call failed for chunk {chunk_idx}: {e}")
                    
                    # Fallback for wav2vec2 style
                    if asr_processor is not None:
                        inputs = asr_processor(
                            chunk.squeeze(0).cpu().numpy(),
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_values.to(device)
                        
                        logits = asr_model(inputs).logits
                        pred_ids = torch.argmax(logits, dim=-1)
                        return asr_processor.batch_decode(pred_ids)[0]
                    
                    return f"[Chunk {chunk_idx}]"
        
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} processing error: {e}")
        return f"[Error chunk {chunk_idx}]"
    finally:
        torch.cuda.empty_cache()

# ==================== MAIN TRANSCRIPTION ====================
@safe_function
def transcribe_with_features(
    audio_path: str,
    language_name: str,
    chunk_length: int = 20,
    enable_diarization: bool = True,
    enable_translation: bool = True,
    progress_callback=None
) -> Tuple[str, str, str, str]:
    """Main transcription function with all features"""
    global asr_model
    
    if audio_path is None:
        return "Please upload audio", "", "", ""
    
    if asr_model is None:
        return "Model not loaded. Please wait or restart.", "", "", ""
    
    lang_code = LANGUAGE_NAME_TO_CODE[language_name]
    logger.info(f"Starting transcription for {language_name} ({lang_code})")
    
    try:
        # Load audio
        if progress_callback:
            progress_callback("Loading audio...")
        
        waveform, sr = torchaudio.load(audio_path)
        logger.info(f"Audio loaded: {waveform.shape}, {sr}Hz")
        
        # Preprocessing
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        duration = waveform.shape[-1] / 16000
        logger.info(f"Audio duration: {duration:.1f}s")
        
        # Diarization
        speaker_segments = []
        if enable_diarization:
            if progress_callback:
                progress_callback("Detecting speakers...")
            speaker_segments = simple_vad_diarization(waveform, 16000)
            logger.info(f"Found {len(speaker_segments)} speaker segments")
        else:
            speaker_segments = [(0, duration, "Speaker_1")]
        
        # Chunking
        chunk_samples = chunk_length * 16000
        chunks = []
        
        for i in range(0, waveform.shape[-1], chunk_samples):
            end = min(i + chunk_samples, waveform.shape[-1])
            chunk = waveform[..., i:end]
            chunks.append((chunk, i/16000, end/16000))
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        # Process chunks
        transcriptions = []
        for idx, (chunk, start, end) in enumerate(chunks):
            if progress_callback:
                progress_callback(f"Processing chunk {idx+1}/{len(chunks)}")
            
            text = process_audio_chunk(chunk, lang_code, idx+1, len(chunks))
            
            # Find speaker
            speaker = "Speaker_1"
            for seg_start, seg_end, seg_speaker in speaker_segments:
                if seg_start <= start < seg_end:
                    speaker = seg_speaker
                    break
            
            transcriptions.append(f"{speaker}: {text}")
        
        # Combine results
        full_transcription = "\n".join(transcriptions)
        
        # Translation
        english_translation = ""
        if enable_translation:
            if progress_callback:
                progress_callback("Translating...")
            
            text_only = " ".join([t.split(": ", 1)[1] if ": " in t else t 
                                for t in transcriptions])
            english_translation = translate_text_simple(text_only)
        
        # Diarization info
        diarization_info = ""
        if enable_diarization and speaker_segments:
            speaker_times = {}
            for start, end, speaker in speaker_segments:
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += (end - start)
            
            diarization_info = f"Speakers: {len(speaker_times)}\n"
            for speaker, time in speaker_times.items():
                diarization_info += f"{speaker}: {time:.1f}s ({time/duration*100:.1f}%)\n"
        
        status = f"✅ Processed {len(chunks)} chunks | Duration: {duration:.1f}s"
        logger.info("Transcription completed successfully")
        
        return full_transcription, english_translation, diarization_info, status
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)[:200]}", "", "", "Failed"

# ==================== GRADIO INTERFACE ====================
def transcribe_wrapper(audio, language, enable_diarization, enable_translation, progress=gr.Progress()):
    """Wrapper for Gradio with progress"""
    try:
        progress(0, desc="Starting...")
        
        def update_progress(msg):
            progress(0.5, desc=msg)
        
        result = transcribe_with_features(
            audio, language, 20, enable_diarization, 
            enable_translation, update_progress
        )
        
        progress(1.0, desc="Complete!")
        return result
        
    except Exception as e:
        logger.error(f"Wrapper error: {e}")
        return f"Error: {str(e)}", "", "", "Failed"

@safe_function
def get_model_status():
    """Get current model status"""
    global model_load_status
    return model_load_status

@safe_function
def clear_cache_handler():
    """Clear cache and return status"""
    try:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cache cleared")
        return check_memory(), "", "", "", "✅ Cache cleared!"
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return check_memory(), "", "", "", f"Error: {str(e)}"

# ==================== INITIALIZE MODELS ====================
def initialize_models():
    """Initialize all models with error handling"""
    global model_load_status
    
    try:
        logger.info("Initializing models...")
        model_load_status = "Initializing..."
        
        # Load ASR
        success = load_asr_model_robust()
        
        # Load translation
        load_translation_model_simple()
        
        if success:
            logger.info("All models initialized successfully")
            model_load_status = "✅ Ready!"
        else:
            logger.warning("Model initialization incomplete")
            model_load_status = "⚠️ Partial load"
        
        return success
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        model_load_status = f"❌ Error: {str(e)[:50]}"
        return False

# ==================== CREATE UI ====================
with gr.Blocks(title="ASR System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ ASR with Diarization & Translation
    
    ⚠️ **Important**: Model loading takes 2-3 minutes. Check status below.
    """)
    
    # Status row with error handling
    with gr.Row():
        with gr.Column(scale=2):
            model_status = gr.Textbox(
                label="Model Status",
                value=get_model_status(),
                interactive=False
            )
        with gr.Column(scale=2):
            memory_status = gr.Textbox(
                label="Memory Status",
                value=check_memory(),
                interactive=False
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
    
    # Log viewer
    with gr.Accordion("📋 System Logs", open=False):
        log_display = gr.Textbox(
            label="Recent Logs",
            value=f"Log file: {log_file}\nSystem ready.",
            lines=5,
            max_lines=10,
            interactive=False
        )
        
        def get_recent_logs():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    return ''.join(lines[-20:])  # Last 20 lines
            except:
                return "Log file not accessible"
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            
            language = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGE_NAME_TO_CODE.keys()),
                value="Hindi"
            )
            
            with gr.Row():
                enable_diarization = gr.Checkbox(
                    label="Speaker Diarization",
                    value=True
                )
                enable_translation = gr.Checkbox(
                    label="Translate to English",
                    value=True
                )
            
            process_btn = gr.Button("🚀 Process Audio", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
            
        with gr.Column():
            status = gr.Textbox(label="Processing Status", lines=2)
            diarization_info = gr.Textbox(label="Speaker Info", lines=3)
            
            with gr.Tab("Transcription"):
                transcription = gr.Textbox(
                    label="Original",
                    lines=10,
                    show_copy_button=True
                )
            
            with gr.Tab("Translation"):
                translation = gr.Textbox(
                    label="English",
                    lines=10,
                    show_copy_button=True
                )
    
    # Event handlers with error protection
    def safe_refresh():
        try:
            return get_model_status(), check_memory(), get_recent_logs()
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            return "Error", "Error", str(e)
    
    refresh_btn.click(
        fn=safe_refresh,
        outputs=[model_status, memory_status, log_display]
    )
    
    process_btn.click(
        fn=transcribe_wrapper,
        inputs=[audio_input, language, enable_diarization, enable_translation],
        outputs=[transcription, translation, diarization_info, status]
    )
    
    clear_btn.click(
        fn=clear_cache_handler,
        outputs=[memory_status, transcription, translation, diarization_info, status]
    )
    
    # Auto-refresh status on load (with error handling)
    demo.load(
        fn=safe_refresh,
        outputs=[model_status, memory_status, log_display]
    )

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("Starting main execution")
        
        # Initialize models
        logger.info("Initializing models (2-3 minutes)...")
        success = initialize_models()
        
        if success:
            logger.info("✅ System ready!")
        else:
            logger.warning("⚠️ System running with limitations")
        
        # Launch Gradio
        logger.info("Launching Gradio interface...")
        demo.queue(max_size=3).launch(
            share=True,
            debug=True,  # Disable debug to avoid verbose output
            show_error=True
        )
        
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        logger.critical(traceback.format_exc())
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print(f"Check log file: {log_file}")
        print("\nTry: Runtime → Restart runtime")
