import os
import subprocess
import tempfile
from embedding import embed_wav_path, cosine_sim


def extract_segment(audio_path, start_time, end_time, output_path):
    """
    Extract audio segment using ffmpeg.
    
    Args:
        audio_path: Path to source audio
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            '-y',
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"Error extracting segment: {e}")
        return False


def match_segments(audio_path, target_embedding, segments, similarity_threshold=0.68):
    """
    Match audio segments against target embedding.
    
    Args:
        audio_path: Path to mixture audio
        target_embedding: Target speaker embedding
        segments: List of (start_time, end_time) tuples
        similarity_threshold: Label threshold for "Target" classification
    
    Returns:
        List of dicts with keys: start, end, similarity, label, segment_path
    """
    matched_segments = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (start, end) in enumerate(segments):
            segment_path = os.path.join(tmpdir, f"segment_{idx}.wav")
            
            if not extract_segment(audio_path, start, end, segment_path):
                continue
            
            try:
                segment_embedding = embed_wav_path(segment_path)
                similarity = cosine_sim(target_embedding, segment_embedding)
                label = "Target" if similarity >= similarity_threshold else "Other"
                
                matched_segments.append({
                    "start": float(start),
                    "end": float(end),
                    "similarity": float(similarity),
                    "label": label,
                    "segment_path": segment_path
                })
            except Exception as e:
                print(f"Error processing segment {idx}: {e}")
                continue
    
    return matched_segments
