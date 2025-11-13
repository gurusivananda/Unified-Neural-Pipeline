import numpy as np
import soundfile as sf

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False


def load_audio(audio_path, sample_rate=16000):
    """Load audio file and resample to 16 kHz."""
    audio, sr = sf.read(audio_path)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio, sample_rate


def detect_speech_segments_webrtc(audio_path, aggressiveness=2, frame_duration_ms=30, min_duration=0.2, merge_threshold=0.2):
    """
    Detect speech segments using WebRTC VAD.
    
    Args:
        audio_path: Path to audio file
        aggressiveness: VAD aggressiveness (0-3)
        frame_duration_ms: Frame duration (10, 20, or 30 ms)
        min_duration: Minimum segment duration in seconds
        merge_threshold: Merge segments closer than this (in seconds)
    
    Returns:
        List of tuples: [(start_time, end_time), ...]
    """
    audio, sample_rate = load_audio(audio_path)
    
    vad = webrtcvad.Vad(aggressiveness)
    
    frame_size = int(sample_rate * frame_duration_ms / 1000.0)
    frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
    
    is_speech = []
    for frame in frames:
        if len(frame) == frame_size:
            frame_bytes = (frame * 32767).astype(np.int16).tobytes()
            speech_detected = vad.is_speech(frame_bytes, sample_rate)
            is_speech.append(speech_detected)
    
    segments = []
    segment_start = None
    
    for i, speech in enumerate(is_speech):
        if speech and segment_start is None:
            segment_start = i
        elif not speech and segment_start is not None:
            segment_end = i
            start_time = segment_start * frame_duration_ms / 1000.0
            end_time = segment_end * frame_duration_ms / 1000.0
            duration = end_time - start_time
            if duration >= min_duration:
                segments.append((start_time, end_time))
            segment_start = None
    
    if segment_start is not None:
        end_time = len(is_speech) * frame_duration_ms / 1000.0
        start_time = segment_start * frame_duration_ms / 1000.0
        duration = end_time - start_time
        if duration >= min_duration:
            segments.append((start_time, end_time))
    
    merged_segments = merge_segments(segments, merge_threshold)
    return merged_segments


def detect_speech_segments_energy(audio_path, energy_threshold=0.02, frame_duration_ms=30, min_duration=0.2, merge_threshold=0.2):
    """
    Simple energy-based speech detection (fallback when WebRTC VAD unavailable).
    
    Args:
        audio_path: Path to audio file
        energy_threshold: Energy threshold for speech detection
        frame_duration_ms: Frame duration in ms
        min_duration: Minimum segment duration in seconds
        merge_threshold: Merge segments closer than this (in seconds)
    
    Returns:
        List of tuples: [(start_time, end_time), ...]
    """
    audio, sample_rate = load_audio(audio_path)
    
    frame_size = int(sample_rate * frame_duration_ms / 1000.0)
    frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
    
    is_speech = []
    for frame in frames:
        if len(frame) == frame_size:
            energy = np.sqrt(np.mean(frame ** 2))
            is_speech.append(energy > energy_threshold)
    
    segments = []
    segment_start = None
    
    for i, speech in enumerate(is_speech):
        if speech and segment_start is None:
            segment_start = i
        elif not speech and segment_start is not None:
            segment_end = i
            start_time = segment_start * frame_duration_ms / 1000.0
            end_time = segment_end * frame_duration_ms / 1000.0
            duration = end_time - start_time
            if duration >= min_duration:
                segments.append((start_time, end_time))
            segment_start = None
    
    if segment_start is not None:
        end_time = len(is_speech) * frame_duration_ms / 1000.0
        start_time = segment_start * frame_duration_ms / 1000.0
        duration = end_time - start_time
        if duration >= min_duration:
            segments.append((start_time, end_time))
    
    merged_segments = merge_segments(segments, merge_threshold)
    return merged_segments


def detect_speech_segments(audio_path, aggressiveness=2, frame_duration_ms=30, min_duration=0.2, merge_threshold=0.2):
    """
    Detect speech segments using WebRTC VAD or energy-based fallback.
    
    Args:
        audio_path: Path to audio file
        aggressiveness: VAD aggressiveness (0-3) - ignored if using energy-based method
        frame_duration_ms: Frame duration (10, 20, or 30 ms)
        min_duration: Minimum segment duration in seconds
        merge_threshold: Merge segments closer than this (in seconds)
    
    Returns:
        List of tuples: [(start_time, end_time), ...]
    """
    if HAS_WEBRTCVAD:
        return detect_speech_segments_webrtc(audio_path, aggressiveness, frame_duration_ms, min_duration, merge_threshold)
    else:
        return detect_speech_segments_energy(audio_path, energy_threshold=0.02, frame_duration_ms=frame_duration_ms, min_duration=min_duration, merge_threshold=merge_threshold)


def merge_segments(segments, threshold=0.2):
    """Merge segments that are closer than threshold seconds."""
    if not segments:
        return []
    
    segments = sorted(segments)
    merged = [list(segments[0])]
    
    for current_start, current_end in segments[1:]:
        last_start, last_end = merged[-1]
        if current_start - last_end < threshold:
            merged[-1][1] = current_end
        else:
            merged.append([current_start, current_end])
    
    return [tuple(seg) for seg in merged]
