import os
import json
import wave
import torch
import pygame
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from lively import LivelyVoice  # Assuming 'lively' is the library for TTS

# Fix numpy error on some platforms
np.core._ARRAY_API = np.__dict__

# Configuration (UPDATE PATHS if needed)
PATHS = {
    'model': "/home/hadisab/Documents/EV/LLM",  # Update paths as required
    'character': "/home/hadisab/Documents/EV/Eve.json",  # Path to your character JSON
    'tts_model': "/home/hadisab/Documents/EV/en_GB-southern_english_female-low.onnx"  # Update to your TTS model path
}

MAX_CONTEXT_LENGTH = 512
MAX_NEW_TOKENS = 120


def load_character(file_path):
    """Load character info from a JSON file."""
    if not os.path.exists(file_path):
        print("Warning: Character file not found. Using defaults.")
        return {
            'name': 'Eve',
            'greeting': 'Hello!',
            'persona': '',
            'example_dialogue': ''
        }
    with open(file_path, 'r') as f:
        char_data = json.load(f)
    return {
        'name': char_data.get('char_name', 'Eve'),
        'greeting': char_data.get('char_greeting', 'Hello!'),
        'persona': char_data.get('char_persona', ''),
        'example_dialogue': char_data.get('example_dialogue', '')
    }


def initialize_tts():
    """Initialize Lively TTS engine."""
    voice = LivelyVoice.load(PATHS['tts_model'])  # Replace with the actual Lively TTS library
    pygame.mixer.init()  # Initialize Pygame mixer for audio playback
    return voice


def synthesize_speech(voice, text):
    """Convert text to speech using Lively."""
    try:
        output_file = "temp_response.wav"

        # Save synthesized speech as a WAV file
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # Standard 16-bit audio
            wav_file.setframerate(22050)  # 22.05 kHz sample rate
            voice.synthesize(text, wav_file)  # Use Lively TTS engine to generate speech

        # Play the generated audio
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Keep the music playing
    except Exception as e:
        print(f"TTS Error: {str(e)}")


def initialize_model(model_path):
    """Initialize the Hugging Face model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded in full precision. Using {device.upper()}.")
    return tokenizer, model, device


def safe_generate(prompt, tokenizer, model, device, character):
    """Generate model response safely with error handling."""
    formatted_prompt = (
        f"Character: {character['name']}\n"
        f"Personality: {character['persona']}\n"
        f"Example Dialogue:\n{character['example_dialogue']}\n"
        f"---\nYou: {prompt}\n{character['name']}:"  # Format input for the model
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=MAX_CONTEXT_LENGTH,
        truncation=True,
        padding=True
    ).to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_k=30,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs.attention_mask.sum().item()
        generated_ids = outputs[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    except Exception as e:
        return f"Let's discuss something else! (Error: {str(e)})"


def main():
    """Main program logic."""
    tokenizer, model, device = initialize_model(PATHS['model'])
    character = load_character(PATHS['character'])
    tts_voice = initialize_tts()

    # Greet the user with character's greeting
    print(f"\n{character['name']}: {character['greeting']}\n")
    synthesize_speech(tts_voice, character['greeting'])

    try:
        while True:
            user_input = input("You: ").strip()  # Wait for user input
            if user_input.lower() in ["quit", "exit"]:
                break

            response = safe_generate(user_input, tokenizer, model, device, character)
            print(f"\n{character['name']}: {response}\n")
            synthesize_speech(tts_voice, response)  # Synthesize speech for the response

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    try:
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)
        print(f"Available RAM: {total_ram:.1f}GB")
    except AttributeError:
        print("Could not determine available RAM on this system.")
    main()
