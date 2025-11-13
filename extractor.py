import os
import subprocess
import tempfile


def concatenate_audio_segments(segment_paths, output_path):
    """
    Concatenate multiple audio files using ffmpeg concat demuxer.
    
    Args:
        segment_paths: List of audio file paths to concatenate
        output_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    if not segment_paths:
        print("No segments to concatenate")
        return False
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in segment_paths:
                escaped_path = path.replace("'", "'\\''")
                f.write(f"file '{path}'\n")
            concat_file = f.name
        
        try:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                '-y',
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        finally:
            os.unlink(concat_file)
    
    except Exception as e:
        print(f"Error concatenating audio: {e}")
        return False
