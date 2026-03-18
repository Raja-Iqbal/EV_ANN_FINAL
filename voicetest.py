from piper import PiperVoice
import pygame  # For audio playback
import wave    # For WAV file handling
import time

VOICE_MODEL = "en_GB-southern_english_female-low.onnx"
voice = PiperVoice.load(VOICE_MODEL)

# Initialize Pygame mixer
pygame.mixer.init()

def speak(text, emotion="neutral"):
    """
    Synthesize speech with emotion and play it using Pygame.
    """
    # Map emotions to SSML prosody
    ssml_profiles = {
        "happy": '<prosody rate="fast" pitch="+20%">',
        "sad": '<prosody rate="slow" pitch="-15%">',
        "angry": '<prosody rate="medium" pitch="+10%" volume="loud">',
        "neutral": ''
    }
    ssml = f'<speak>{ssml_profiles[emotion]}{text}</prosody></speak>'
    
    # Synthesize speech to a temporary WAV file
    output_file = "temp.wav"
    with wave.open(output_file, "wb") as wav_file:
        # Set WAV file parameters
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(22050)  # Sample rate (adjust if needed)
        voice.synthesize(ssml, wav_file)  # Piper auto-detects SSML
    
    # Play the audio using Pygame
    pygame.mixer.music.load(output_file)
    pygame.mixer.music.play()
    
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Test emotions
speak("I'm really excited to talk to you!", emotion="happy")
speak("This makes me a bit sad...", emotion="sad")
speak("I'm feeling neutral about this.", emotion="neutral")
speak("This is making me angry!", emotion="angry")
