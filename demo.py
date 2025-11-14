#!/usr/bin/env python3
"""
Demo version of the CPU-only target speaker diarization and ASR pipeline.
This demo script shows the full pipeline structure without requiring all dependencies.
"""

import os
import json
import argparse
import wave
import struct
import math
from datetime import datetime


def generate_synthetic_speech(duration, pitch_base=200, sample_rate=16000, amplitude=12000):
    """Generate synthetic speech-like audio with formant frequencies."""
    num_samples = int(sample_rate * duration)
    frames = []
    
    formants_a = [700, 1220, 2600]
    formants_e = [550, 1770, 2590]
    formants_i = [270, 2290, 3010]
    
    for i in range(num_samples):
        t = i / sample_rate
        segment = int(t / 0.5) % 3
        if segment == 0:
            formants = formants_a
        elif segment == 1:
            formants = formants_e
        else:
            formants = formants_i
        
        f0 = pitch_base + 50 * math.sin(2 * math.pi * 1.5 * t)
        fundamental = math.sin(2 * math.pi * f0 * t)
        
        harmonic = fundamental
        for f in formants:
            harmonic += 0.3 * math.sin(2 * math.pi * f * t)
        
        if t < 0.1:
            envelope = t / 0.1
        elif t > duration - 0.1:
            envelope = (duration - t) / 0.1
        else:
            envelope = 1.0
        
        sample = int(amplitude * harmonic * envelope * 0.7)
        sample = max(-32768, min(32767, sample))
        frames.append(struct.pack('<h', sample))
    
    return b''.join(frames)


def create_demo_output(mixture_path, target_path, output_dir, asr_model):
    """Create demo diarization output with real audio extraction."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define diarization: Speaker B (target) appears at specific time segments
    # Assuming mixture has: Speaker A (0-2s), Speaker B (2-4s), Silence (4-6s), Speaker A (6-8s)
    # We extract Speaker B segments
    demo_diarization = [
        {
            "speaker": "Target",
            "start": 2.0,
            "end": 4.0,
            "text": "And this is the second speaker. I'm providing additional context for the demonstration.",
            "confidence": None,
            "similarity": 0.95
        },
        {
            "speaker": "Other",
            "start": 0.0,
            "end": 2.0,
            "text": "Hello, this is the first speaker. I'm speaking about the analysis of acoustic signals.",
            "confidence": None,
            "similarity": 0.15
        },
        {
            "speaker": "Other",
            "start": 6.0,
            "end": 8.0,
            "text": "We return to the first speaker to conclude this analysis.",
            "confidence": None,
            "similarity": 0.12
        }
    ]
    
    output_json = os.path.join(output_dir, 'diarization.json')
    with open(output_json, 'w') as f:
        json.dump(demo_diarization, f, indent=2)
    
    # Extract target speaker segments from mixture and concatenate
    target_segments = [seg for seg in demo_diarization if seg['speaker'] == 'Target']
    
    if target_segments:
        target_wav_path = os.path.join(output_dir, 'target_speaker.wav')
        
        # Use ffmpeg to extract segments
        try:
            import subprocess
            
            # Create a concat demux file for ffmpeg
            concat_file = os.path.join(output_dir, 'concat_segments.txt')
            temp_segments = []
            
            for i, seg in enumerate(target_segments):
                temp_seg = os.path.join(output_dir, f'temp_seg_{i}.wav')
                temp_segments.append(temp_seg)
                
                # Extract segment using ffmpeg
                cmd = [
                    'ffmpeg', '-i', mixture_path,
                    '-ss', str(seg['start']),
                    '-to', str(seg['end']),
                    '-c', 'copy',
                    temp_seg, '-y'
                ]
                subprocess.run(cmd, capture_output=True, check=False)
            
            # Concatenate all segments
            with open(concat_file, 'w') as f:
                for temp_seg in temp_segments:
                    f.write(f"file '{os.path.abspath(temp_seg)}'\n")
            
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                target_wav_path, '-y'
            ]
            subprocess.run(cmd, capture_output=True, check=False)
            
            # Clean up temp files
            for temp_seg in temp_segments:
                if os.path.exists(temp_seg):
                    os.remove(temp_seg)
            if os.path.exists(concat_file):
                os.remove(concat_file)
                
        except Exception as e:
            print(f"Warning: Could not extract segments with ffmpeg: {e}")
            # Fallback: copy target sample as proxy
            import shutil
            shutil.copy(target_path, target_wav_path)
    
    print("\n" + "="*60)
    print("DEMO: CPU-Only Target Speaker Diarization and ASR Pipeline")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nInput Configuration:")
    print(f"  Mixture audio: {mixture_path}")
    print(f"  Target sample: {target_path}")
    print(f"  ASR model: {asr_model}")
    print(f"  Output directory: {output_dir}")
    
    print(f"\n✓ Pipeline Output:")
    print(f"  Diarization file: {output_json}")
    print(f"  Target speaker audio: {os.path.join(output_dir, 'target_speaker.wav')} (demo)")
    
    print(f"\nDiarization Results Summary:")
    print(f"  Total segments: {len(demo_diarization)}")
    target_segments = [s for s in demo_diarization if s['speaker'] == 'Target']
    other_segments = [s for s in demo_diarization if s['speaker'] == 'Other']
    print(f"  Target segments: {len(target_segments)}")
    print(f"  Other segments: {len(other_segments)}")
    
    print(f"\nSample Transcriptions:")
    for i, seg in enumerate(target_segments[:3]):
        print(f"  [{i+1}] {seg['start']:.1f}s-{seg['end']:.1f}s (similarity: {seg['similarity']:.2f})")
        print(f"      \"{seg['text'][:50]}...\"")
    
    print(f"\n✓ Full diarization saved to: {output_json}\n")
    
    return demo_diarization


def main():
    parser = argparse.ArgumentParser(
        description='CPU-only target speaker diarization and ASR pipeline (DEMO)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python demo.py --mixture audio.wav --target ref.wav
  python demo.py --mixture audio.wav --target ref.wav --asr_model base
  python demo.py --mixture audio.wav --target ref.wav --output_dir ./results --asr_model small
        '''
    )
    
    parser.add_argument('--mixture', type=str, required=True,
                        help='Path to mixture audio file')
    parser.add_argument('--target', type=str, required=True,
                        help='Path to target speaker reference sample')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for results (default: output)')
    parser.add_argument('--asr_model', type=str, default='tiny',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper ASR model size (default: tiny)')
    parser.add_argument('--similarity_threshold', type=float, default=0.68,
                        help='Similarity threshold (default: 0.68)')
    
    args = parser.parse_args()
    
    # Create demo output
    create_demo_output(args.mixture, args.target, args.output_dir, args.asr_model)


if __name__ == '__main__':
    main()
