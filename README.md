# Target Speaker Diarization and ASR Pipeline

A complete Python pipeline for speaker diarization and automatic speech recognition (ASR) optimized for CPU-only execution. This system identifies and transcribes a target speaker from multi-speaker audio using webrtcvad for voice activity detection, resemblyzer for speaker embeddings, and OpenAI's Whisper for ASR.

## Features

- **Speech Segmentation**: WebRTC VAD (Voice Activity Detection) for CPU-efficient speech segment detection
- **Speaker Identification**: Resemblyzer-based speaker embeddings with cosine similarity matching
- **Audio Extraction**: ffmpeg-based efficient audio segment extraction and concatenation
- **ASR**: OpenAI Whisper (tiny/base/small/medium/large models) on CPU
- **Complete Pipeline**: End-to-end diarization with speaker labels and timestamps

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

The pipeline requires ffmpeg to be installed on your system:
- **Linux**: `apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`

## Usage

### Basic Command

```bash
python main.py --mixture mixture_audio.wav --target target_sample.wav --output_dir output --asr_model tiny
```

### Arguments

- `--mixture` (required): Path to multi-speaker audio file (WAV, MP3, etc.)
- `--target` (required): Path to 3-10 second reference sample of target speaker
- `--output_dir` (default: `output`): Directory for output files
- `--asr_model` (default: `tiny`): Whisper model size
  - `tiny`: ~39M parameters, fastest
  - `base`: ~74M parameters
  - `small`: ~244M parameters, best quality on CPU
  - `medium`: ~769M parameters
  - `large`: ~1550M parameters
- `--similarity_threshold` (default: `0.68`): Cosine similarity threshold for target speaker classification

### Example Workflows

**Fast processing (tiny model, threshold 0.70):**
```bash
python main.py --mixture audio.wav --target reference.wav --asr_model tiny --similarity_threshold 0.70
```

**High quality (small model, threshold 0.65):**
```bash
python main.py --mixture audio.wav --target reference.wav --asr_model small --similarity_threshold 0.65
```

**Custom output directory:**
```bash
python main.py --mixture audio.wav --target reference.wav --output_dir ./results
```

## Output

The pipeline generates:

1. **`output/target_speaker.wav`**: Concatenated audio of all detected target speaker segments
2. **`output/diarization.json`**: Speaker diarization results with timestamps and transcriptions

### Output Format (diarization.json)

```json
[
  {
    "speaker": "Target",
    "start": 0.5,
    "end": 3.2,
    "text": "Hello, this is the target speaker.",
    "confidence": null,
    "similarity": 0.72
  },
  ...
]
```

- `speaker`: Speaker label ("Target" for identified target speaker)
- `start`: Segment start time in seconds
- `end`: Segment end time in seconds
- `text`: Transcribed text from Whisper
- `confidence`: Reserved for future use (currently null)
- `similarity`: Cosine similarity score to target speaker embedding (0-1)

## Tuning the Similarity Threshold

The `--similarity_threshold` parameter controls how strictly the system classifies segments as the target speaker:

- **Higher threshold (0.70-0.75)**: More conservative, fewer false positives but may miss target segments
- **Default (0.68)**: Balanced classification
- **Lower threshold (0.60-0.65)**: More aggressive, may include non-target speakers

### Recommendation for Tuning

1. Start with the default threshold (0.68)
2. Check the similarities in the output JSON
3. If too many "Other" speakers are included, increase the threshold
4. If target speaker segments are missing, decrease the threshold
5. Run with `--similarity_threshold` to adjust

## How It Works

### Pipeline Steps

1. **Compute Target Embedding**: Resemblyzer VoiceEncoder processes the target sample to create a 256-dimensional speaker embedding
2. **Speech Segmentation**: WebRTC VAD detects speech segments in the mixture audio with 30ms frame windows
3. **Segment Matching**: For each detected segment:
   - Extract using ffmpeg
   - Compute embedding via Resemblyzer
   - Calculate cosine similarity to target embedding
   - Label as "Target" (≥ 0.68) or "Other"
4. **Audio Concatenation**: Ffmpeg concatenates all target segments
5. **ASR**: Whisper transcribes each target segment
6. **Output**: Generate JSON with speaker, timing, and transcription

### CPU Optimization

- Webrtcvad: Highly optimized C++ VAD, minimal CPU usage
- Resemblyzer: Efficient neural network inference on CPU
- Whisper: Supports CPU inference with quantization support
- Ffmpeg: Hardware-accelerated where available, falls back to CPU

## Module Overview

- `main.py`: Main pipeline orchestrator
- `diarization.py`: WebRTC VAD speech segmentation
- `embedding.py`: Resemblyzer speaker embedding computation
- `matcher.py`: Segment-to-speaker matching with similarity scoring
- `extractor.py`: ffmpeg-based audio segment extraction and concatenation
- `asr.py`: Whisper ASR inference
- `utils.py`: Helper functions for file I/O and JSON serialization

## Performance Notes

- **Tiny Model**: ~10-30 seconds per minute of audio on modern CPU
- **Base Model**: ~30-60 seconds per minute of audio
- **Small Model**: ~1-2 minutes per minute of audio
- **VAD**: ~0.5-1 second per minute of audio

Actual performance depends on CPU speed and system load.

## Troubleshooting

**ImportError for resemblyzer/webrtcvad**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Some packages may require build tools on Windows

**ffmpeg not found**
- Install ffmpeg system-wide and ensure it's in PATH
- On Windows, test with `ffmpeg -version` in PowerShell

**No target segments detected**
- Lower the `--similarity_threshold` (try 0.60-0.65)
- Ensure target sample is clear speech of the same speaker
- Verify audio formats are supported

**Out of memory on large files**
- Use smaller Whisper models (tiny or base)
- Consider splitting long audio files into chunks
- Close other applications to free memory

## Dependencies

All dependencies are CPU-compatible:
- `webrtcvad==4.3.1`: Voice Activity Detection
- `resemblyzer==0.1.1.dev0`: Speaker embeddings
- `openai-whisper==20230314`: Speech-to-text
- `numpy==1.24.3`: Numerical computing
- `scipy==1.10.1`: Scientific computing
- `librosa==0.10.0`: Audio processing
- `soundfile==0.12.1`: Audio file I/O

## License

This project is provided as-is for research and development purposes.
