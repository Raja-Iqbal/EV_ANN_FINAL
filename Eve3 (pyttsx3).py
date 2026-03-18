import numpy as np
import cv2
import os
import torch
import json
import pyttsx3
import wave
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil
from collections import deque


# Paths to resources
PATHS = {
    'model': r"C:\PY PROJECTS\EV\LLM",
    'character': r"C:\PY PROJECTS\EV\Eve.json"
}

MAX_HISTORY = 6  # Number of lines of dialogue to keep
MAX_NEW_TOKENS = 120
PRECISION_MODE = '8bit'


def load_character(file_path):
    try:
        with open(file_path, 'r') as f:
            char_data = json.load(f)
        return {
            'name': char_data.get('char_name', 'Eve'),
            'greeting': char_data.get('char_greeting', 'Hello!'),
            'persona': char_data.get('char_persona', ''),
        }
    except:
        return {'name': 'Eve', 'greeting': 'Hello!', 'persona': ''}


def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    return engine


def speak(engine, text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def initialize_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.float32 if PRECISION_MODE == '8bit' else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def format_prompt(character_name, history):
    return "\n".join(history) + f"\n{character_name}:"


def safe_generate(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_k=40,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result[len(prompt):].split("\n")[0].strip()
    except Exception as e:
        return f"(Error generating response: {e})"


def main():
    print(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        print(f"CUDA is available. Running on {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU.")

    tokenizer, model = initialize_model(PATHS['model'])
    character = load_character(PATHS['character'])
    tts = initialize_tts()

    history = deque(maxlen=MAX_HISTORY)
    print(f"\n{character['name']}: {character['greeting']}\n")
    speak(tts, character['greeting'])
    history.append(f"{character['name']}: {character['greeting']}")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            history.append(f"User: {user_input}")
            prompt = format_prompt(character['name'], list(history))
            response = safe_generate(prompt, tokenizer, model)

            if not response:
                response = "I'm not sure what to say to that."

            print(f"\n{character['name']}: {response}\n")
            speak(tts, response)
            history.append(f"{character['name']}: {response}")

    except KeyboardInterrupt:
        print("\nExited by user.")


if __name__ == "__main__":
    main()
