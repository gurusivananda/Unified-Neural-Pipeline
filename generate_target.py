import pyttsx3
import os

OUTPUT = 'target_sample.wav'

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Slower speech for clarity

# Use a female voice for the target speaker
voices = engine.getProperty('voices')
print(f"Available voices: {len(voices)}")
for i, v in enumerate(voices):
    print(f"  {i}: {v.name}")

# Select female voice (usually index 1)
female_voice = voices[1].id if len(voices) > 1 else voices[0].id
engine.setProperty('voice', female_voice)

print("Generating target speaker sample (female voice)...")
engine.save_to_file(
    "This is the target speaker. My voice should be consistently recognized and separated from other speakers in the mixture.",
    OUTPUT
)
engine.runAndWait()

print(f"✓ Wrote {OUTPUT} with real human speech (female voice)")

