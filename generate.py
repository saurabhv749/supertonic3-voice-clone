import argparse
import os
import time
import soundfile as sf
from helper import load_text_to_speech, timer, load_voice_style

save_dir = "samples"
os.makedirs(save_dir, exist_ok=True)

tts = load_text_to_speech('supertonic3/onnx')

def argparse_args():
    parser = argparse.ArgumentParser(description="Generate speech from text using trained voice style.")
    parser.add_argument("--text", type=str, required=True, help="Input text to synthesize.")
    parser.add_argument("--lang", type=str, default="en", help="Language code (e.g., 'en' for English).")
    parser.add_argument("--style", type=str, required=True, help="Path to the voice style JSON file.")
    parser.add_argument("--total_step", type=int, default=6, help="Total steps for generation.")
    parser.add_argument("--speed", type=float, default=1.05, help="Speed factor for the generated speech.")
    return parser.parse_args()

args = argparse_args()
style_path = args.style
text = args.text
lang = args.lang
total_step = args.total_step
speed = args.speed

voice_style = load_voice_style([style_path], verbose=True)
voice_name = os.path.basename(style_path).replace(".json", "")
print(f"\n=== Generating speech for text: '{text}' with voice style: '{voice_name}' ===")

with timer("Generating speech from text"):
    wav, duration = tts(
        text, 
        lang,
        voice_style, 
        total_step, 
        speed
        )
# get current time
timestamp = int(time.time())
fname = f"{voice_name}_{timestamp}.wav"
w = wav[0, : int(tts.sample_rate * duration[0].item())]  # [T_trim]
sf.write(os.path.join(save_dir, fname), w, tts.sample_rate)
print("\n=== Synthesis completed successfully! ===")
print(f"Saved: {save_dir}/{fname}")