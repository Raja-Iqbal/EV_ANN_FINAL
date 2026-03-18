import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import simpleaudio as sa

# === CONFIGURATION ===
PATHS = {
    'model': r"C:\PY PROJECTS\EV\LLM",        # Path to your LLM model
    'character': r"C:\PY PROJECTS\EV\Eve.json"  # Path to character JSON file
}

MAX_CONTEXT_LENGTH = 512
MAX_NEW_TOKENS = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD CHARACTER INFO ===
def load_character(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            char_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Character file not found at {file_path}, using defaults.")
        return {
            'name': 'Eve',
            'greeting': "Hey there! I'm Eve. What would you like to chat about today?",
            'persona': (
                "You are Eve, a warm, empathetic, and witty AI companion. "
                "You speak like a close friend who is curious, kind, and full of gentle humor. "
                "Keep replies light, positive, and human-like. Avoid controversial or heavy topics.\n"
                "Example dialogue:\n"
                "User: Hi Eve!\n"
                "Eve: Hey! So glad to hear from you. What’s been the highlight of your day so far?"
            ),
            'example_dialogue': ''
        }
    else:
        return {
            'name': char_data.get('char_name', 'Eve'),
            'greeting': char_data.get('char_greeting', "Hey there! I'm Eve. What would you like to chat about today?"),
            'persona': char_data.get('char_persona', ''),
            'example_dialogue': char_data.get('example_dialogue', '')
        }

# === INITIALIZE MODEL & TOKENIZER ===
def initialize_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# === SAFE TEXT GENERATION WITH TOPIC FILTERING AND LENGTH CONTROL ===
def safe_generate(prompt, tokenizer, model):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_CONTEXT_LENGTH,
        truncation=True,
        padding=True
    ).to(DEVICE)

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

        input_len = inputs['attention_mask'].sum().item()
        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Remove any repeated prompt text
        if response.lower().startswith(prompt.lower()):
            response = response[len(prompt):].strip()

        # Basic sensitive topic detection and redirection
        sensitive_topics = ['abuse', 'violence', 'suicide', 'hate', 'racism', 'sex', 'fetish', 'porn']
        if any(topic in response.lower() for topic in sensitive_topics):
            return ("That’s a sensitive topic. I’m here to keep things positive and friendly. "
                    "Let’s chat about something fun or interesting instead! 😊")

        # Limit response length for natural flow
        max_len_chars = 180
        if len(response) > max_len_chars:
            response = response[:max_len_chars].rstrip() + "..."

        # Capitalize first letter for nicer formatting
        response = response[0].upper() + response[1:] if response else response

        return response

    except Exception as e:
        print(f"[Error] Generation failed: {e}")
        return "Oops, something went wrong on my end. Let's try again!"

# === INITIALIZE TTS ENGINE ===
def initialize_tts():
    return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available(), progress_bar=False)

# === SYNTHESIZE AND PLAY AUDIO ===
def synthesize_speech(tts, text):
    try:
        temp_wav = "temp_output.wav"
        tts.tts_to_file(text=text, file_path=temp_wav)

        wave_obj = sa.WaveObject.from_wave_file(temp_wav)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        os.remove(temp_wav)
    except Exception as e:
        print(f"[TTS Playback Error] {e}")

# === MAIN CHAT LOOP ===
def main():
    print(f"Starting chat on device: {DEVICE}\n")

    tokenizer, model = initialize_model(PATHS['model'])
    character = load_character(PATHS['character'])
    tts = initialize_tts()

    # Greet user warmly
    print(f"{character['name']}: {character['greeting']}\n")
    synthesize_speech(tts, character['greeting'])

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print(f"{character['name']}: It was great chatting! Take care! 👋")
                synthesize_speech(tts, "It was great chatting! Take care!")
                break

            # Construct prompt including persona, example dialogue, and user input
            prompt = (
                f"{character['persona']}\n\n"
                f"User: {user_input}\n"
                f"{character['name']}:"
            )

            response = safe_generate(prompt, tokenizer, model)
            print(f"\n{character['name']}: {response}\n")
            synthesize_speech(tts, response)

    except KeyboardInterrupt:
        print(f"\n{character['name']}: Goodbye! Hope to chat again soon. 👋")

if __name__ == "__main__":
    main()
