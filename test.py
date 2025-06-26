import whisper

# Load the base model (can be "tiny", "base", "small", "medium", or "large")
model = whisper.load_model("base",device="cuda",)  # Loads the model (downloads if not present)
# You can also specify a device, e.g., "cuda" for GPU support
# model = whisper.load_model("base", device="cuda")  # Uncomment if you have a GPU

# Transcribe an audio file (must be wav/mp3/m4a/webm etc.)
result = model.transcribe("meeting_audio.wav",
                         language='en',  # Set to None for auto-detection
                         task="transcribe")  # Use "translate" for translation to English
  # Specify the language if known, e.g., "en" for English'
  # Replace with your audio file path

# Print the text
print("\n📄 Transcription:")
print(result["text"])
