#!/usr/bin/env python3
"""
Demo version of the CPU-only target speaker diarization and ASR pipeline.
This demo script shows the full pipeline structure without requiring all dependencies.
"""

import os
import json
import argparse
from datetime import datetime


def create_demo_output(mixture_path, target_path, output_dir, asr_model):
    """Create demo diarization output without running full pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    demo_diarization = [
        {
            "speaker": "Target",
            "start": 0.5,
            "end": 3.2,
            "text": "Hello, this is a demo of the target speaker diarization pipeline.",
            "confidence": None,
            "similarity": 0.82
        },
        {
            "speaker": "Target",
            "start": 5.1,
            "end": 8.7,
            "text": "This demonstrates how the system identifies and transcribes target speakers.",
            "confidence": None,
            "similarity": 0.75
        },
        {
            "speaker": "Other",
            "start": 9.2,
            "end": 12.0,
            "text": "This is a different speaker not matching the target.",
            "confidence": None,
            "similarity": 0.32
        },
        {
            "speaker": "Target",
            "start": 13.5,
            "end": 16.8,
            "text": "The system concatenates all target segments into one audio file.",
            "confidence": None,
            "similarity": 0.79
        }
    ]
    
    output_json = os.path.join(output_dir, 'diarization.json')
    with open(output_json, 'w') as f:
        json.dump(demo_diarization, f, indent=2)
    
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
