import whisper


def transcribe_audio(audio_path, model_name='tiny'):
    """
    Transcribe audio using Whisper ASR (CPU-only).
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        Tuple of (transcription_text, full_result_dict)
    """
    try:
        model = whisper.load_model(model_name, device="cpu")
        result = model.transcribe(audio_path, language="en", device="cpu")
        transcription = result.get("text", "").strip()
        return transcription, result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "", {"text": "", "error": str(e)}


def transcribe_segments(segment_paths, model_name='tiny'):
    """
    Transcribe multiple audio segments.
    
    Args:
        segment_paths: List of audio file paths
        model_name: Whisper model size
    
    Returns:
        List of transcription texts
    """
    transcriptions = []
    for path in segment_paths:
        text, _ = transcribe_audio(path, model_name)
        transcriptions.append(text)
    return transcriptions
