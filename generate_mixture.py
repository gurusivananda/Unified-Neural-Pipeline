import pyttsx3
import os
import wave

OUTPUT = 'mixture_audio.wav'

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Get available voices
voices = engine.getProperty('voices')
print(f"Available voices: {len(voices)}")
for i, v in enumerate(voices):
    print(f"  {i}: {v.name}")

# Select male and female voices
male_voice = voices[0].id if len(voices) > 0 else None
female_voice = voices[1].id if len(voices) > 1 else voices[0].id

print("\nGenerating mixture with both speakers...")

# Generate Speaker A (male) - first part
print("  Generating Speaker A (male)...")
temp_a1 = 'temp_speaker_a1.wav'
engine.setProperty('voice', male_voice)
engine.save_to_file(
    "Hello, this is the first speaker. I'm speaking about the analysis of acoustic signals and speaker recognition.",
    temp_a1
)
engine.runAndWait()

# Generate Speaker B (female) - middle part
print("  Generating Speaker B (female)...")
temp_b = 'temp_speaker_b.wav'
engine.setProperty('voice', female_voice)
engine.save_to_file(
    "And this is the second speaker. I'm providing additional context for the demonstration of target speaker extraction.",
    temp_b
)
engine.runAndWait()

# Generate Speaker A (male) - last part
print("  Generating Speaker A (male) again...")
temp_a2 = 'temp_speaker_a2.wav'
engine.setProperty('voice', male_voice)
engine.save_to_file(
    "We return to the first speaker to conclude this analysis.",
    temp_a2
)
engine.runAndWait()

# Manual concatenation with silence between speakers
print("Concatenating segments...")
all_frames = []
sample_rate = None

# Read Speaker A segment 1
with wave.open(temp_a1, 'rb') as wf:
    sample_rate = wf.getframerate()
    all_frames.append(wf.readframes(wf.getnframes()))

# Add silence (2 seconds)
if sample_rate:
    silence = b'\x00\x00' * (sample_rate * 2)
    all_frames.append(silence)

# Read Speaker B segment
with wave.open(temp_b, 'rb') as wf:
    all_frames.append(wf.readframes(wf.getnframes()))

# Add silence (2 seconds)
if sample_rate:
    all_frames.append(silence)

# Read Speaker A segment 2
with wave.open(temp_a2, 'rb') as wf:
    all_frames.append(wf.readframes(wf.getnframes()))

# Write concatenated mixture
if sample_rate and all_frames:
    with wave.open(OUTPUT, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(all_frames))
    print(f"✓ Wrote {OUTPUT}")

# Clean up temp files
for temp_file in [temp_a1, temp_b, temp_a2]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print(f"✓ Mixture generated: {OUTPUT} with Speaker A (male) + Speaker B (female) voices")


