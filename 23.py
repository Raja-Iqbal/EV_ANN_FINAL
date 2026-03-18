from TTS.api import TTS

# Load the model (GPU will be used automatically if available)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

# Convert text to speech and save to output.wav
tts.tts_to_file(text="Hello, I am your AI assistant!", file_path="output.wav")
