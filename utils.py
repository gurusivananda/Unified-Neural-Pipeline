import os
import json
import subprocess
import tempfile
from pathlib import Path


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def save_diarization_json(diarization_data, output_path):
    """
    Save diarization results to JSON file.
    
    Args:
        diarization_data: List of dicts with speaker, start, end, text, confidence, similarity
        output_path: Output JSON file path
    """
    with open(output_path, 'w') as f:
        json.dump(diarization_data, f, indent=2)


def copy_segment_file(src, dst):
    """
    Copy audio segment file using ffmpeg.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', src,
            '-c', 'copy',
            '-y',
            dst
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False


def cleanup_temp_files(temp_dir):
    """Clean up temporary directory."""
    try:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")
