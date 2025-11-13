import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path

from diarization import detect_speech_segments
from embedding import embed_wav_path
from matcher import match_segments, extract_segment
from extractor import concatenate_audio_segments
from asr import transcribe_audio
from utils import ensure_output_dir, save_diarization_json


def copy_segment_to_temp(src, dst):
    """Copy audio segment using ffmpeg."""
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


def main():
    parser = argparse.ArgumentParser(
        description='CPU-only target speaker diarization and ASR pipeline'
    )
    parser.add_argument('--mixture', type=str, required=True,
                        help='Path to mixture audio file')
    parser.add_argument('--target', type=str, required=True,
                        help='Path to target speaker reference sample')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for results')
    parser.add_argument('--asr_model', type=str, default='tiny',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper ASR model size')
    parser.add_argument('--similarity_threshold', type=float, default=0.68,
                        help='Similarity threshold for target speaker classification')
    
    args = parser.parse_args()
    
    ensure_output_dir(args.output_dir)
    
    print("[1/6] Loading target speaker embedding...")
    try:
        target_embedding = embed_wav_path(args.target)
        print(f"✓ Target embedding computed (shape: {target_embedding.shape})")
    except Exception as e:
        print(f"✗ Error computing target embedding: {e}")
        sys.exit(1)
    
    print("\n[2/6] Detecting speech segments with VAD...")
    try:
        segments = detect_speech_segments(args.mixture, aggressiveness=2)
        print(f"✓ Detected {len(segments)} speech segments")
        for i, (start, end) in enumerate(segments[:5]):
            print(f"  Segment {i}: {start:.2f}s - {end:.2f}s")
        if len(segments) > 5:
            print(f"  ... and {len(segments) - 5} more")
    except Exception as e:
        print(f"✗ Error detecting segments: {e}")
        sys.exit(1)
    
    print("\n[3/6] Matching segments to target speaker...")
    try:
        matched_segments = match_segments(
            args.mixture,
            target_embedding,
            segments,
            similarity_threshold=args.similarity_threshold
        )
        target_segments = [s for s in matched_segments if s['label'] == 'Target']
        other_segments = [s for s in matched_segments if s['label'] == 'Other']
        print(f"✓ Classified {len(target_segments)} target segments, {len(other_segments)} other")
    except Exception as e:
        print(f"✗ Error matching segments: {e}")
        sys.exit(1)
    
    if not target_segments:
        print("✗ No target segments found. Adjust similarity threshold and try again.")
        sys.exit(1)
    
    print("\n[4/6] Extracting and concatenating target segments...")
    with tempfile.TemporaryDirectory() as tmpdir:
        target_wav_path = os.path.join(args.output_dir, 'target_speaker.wav')
        
        try:
            temp_segments = []
            for idx, seg_info in enumerate(target_segments):
                temp_seg = os.path.join(tmpdir, f"target_{idx}.wav")
                if extract_segment(
                    args.mixture,
                    seg_info['start'],
                    seg_info['end'],
                    temp_seg
                ):
                    temp_segments.append(temp_seg)
            
            if concatenate_audio_segments(temp_segments, target_wav_path):
                print(f"✓ Concatenated target segments: {target_wav_path}")
            else:
                print("✗ Error concatenating segments")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Error in extraction: {e}")
            sys.exit(1)
        
        print("\n[5/6] Running ASR on target segments...")
        diarization_results = []
        
        try:
            for idx, seg_info in enumerate(target_segments):
                text, result = transcribe_audio(temp_segments[idx], args.asr_model)
                
                diarization_results.append({
                    "speaker": "Target",
                    "start": seg_info['start'],
                    "end": seg_info['end'],
                    "text": text,
                    "confidence": None,
                    "similarity": seg_info['similarity']
                })
                
                if idx < 3:
                    print(f"  [{idx}] {seg_info['start']:.2f}s-{seg_info['end']:.2f}s: {text[:60]}")
            
            print(f"✓ Transcribed {len(diarization_results)} segments")
        except Exception as e:
            print(f"✗ Error running ASR: {e}")
            sys.exit(1)
        
        print("\n[6/6] Saving results...")
        try:
            diarization_json = os.path.join(args.output_dir, 'diarization.json')
            save_diarization_json(diarization_results, diarization_json)
            print(f"✓ Saved diarization results: {diarization_json}")
            
            print(f"\n✓ Pipeline complete!")
            print(f"  Target audio: {target_wav_path}")
            print(f"  Diarization: {diarization_json}")
            print(f"  Segments processed: {len(diarization_results)}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
