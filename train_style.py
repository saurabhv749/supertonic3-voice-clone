"""
Modified version of 'optimize_style.py' 
Original Author: [Gyeongmin Kim] (https://github.com/kdrkdrkdr)
Maintained by: [Saurabh Verma]


Extract voice style JSON from a WAV file for SupertonicTTS.
Core approach:
  - Convert ONNX TTS models to PyTorch (enables gradient backpropagation)
  - Optimize style vectors via Speaker similarity

Usage:
    python train_style.py
"""

import argparse
import os
import glob
import time
import numpy as np
import torch

from helper import load_text_to_speech, load_voice_style as load_voice_styles
from utils import save_style, SupertonicModel, get_train_dataloader
from configs import texts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser():
    parser = argparse.ArgumentParser(description="Optimize a voice style from a target WAV file")
    parser.add_argument("--name", default="F6", help="Output name for logs and saved style")
    parser.add_argument("--gender", default="F", choices=["F", "M"], help="Voice gender for selecting the default reference style")
    parser.add_argument("--target-wav-path", default="voices/F6.wav", help="Path to the target WAV file")
    parser.add_argument("--reference-style", default="auto", help="Reference style source: 'auto', 'none', or a path to a style checkpoint JSON file")
    parser.add_argument("--seed", type=int, default=49, help="Random seed")
    parser.add_argument("--speed", type=float, default=1.05, help="Speech speed multiplier")
    parser.add_argument("--num-steps", type=int, default=3000, help="Number of optimization steps")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="Optimizer learning rate")
    parser.add_argument("--vocoder-steps", type=int, default=6, help="Vocoder steps used during evaluation")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--early-stop-loss-threshold", type=float, default=0.015, help="Loss threshold for early stopping")
    return parser


def load_voice_style(path):
    """ load single voice style """
    style = load_voice_styles([path]) # Style
    style_ttl = torch.tensor(style.ttl, dtype=torch.float32).to(DEVICE) # (1, 50, 256)
    style_dp = torch.tensor(style.dp, dtype=torch.float32).to(DEVICE) # (1, 8, 16)
    return style_ttl, style_dp


def main():
    # ===== configs =====
    parser = build_parser()
    args = parser.parse_args()
    name = args.name
    gender = args.gender
    target_wav_path = args.target_wav_path
    reference_style = None if args.reference_style == "none" else args.reference_style
    seed = args.seed
    speed = args.speed
    vocoder_steps = args.vocoder_steps
    

    # models
    model = SupertonicModel("supertonic3/onnx", target_wav_path)
    TTS = load_text_to_speech("supertonic3/onnx")

    print(f"Using device: {DEVICE}")
    print(f"Name: {name}")
    
    # Paths
    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # ===== Generate fixed noisy latent (seed-controlled) =====

    # create dataset
    dataloader = get_train_dataloader(TTS, texts)
    data_iter = iter(dataloader)
    # seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Use the first text as a reference for latent length and style extraction
    tmp_input_ids, tmp_attention_mask = next(data_iter)
    # voice style
    tmp_voice = "F4.json" if gender == "F" else "M1.json"
    tmp_voice_path = f"supertonic3/voice_styles/{tmp_voice}"
    tmp_ttl, tmp_dp = load_voice_style(tmp_voice_path) # (1, 50, 256), (1, 8, 16)
    
    
    # latent computation
    with torch.no_grad():
        init_dur = model.dp_model(tmp_input_ids, tmp_dp, tmp_attention_mask) / speed
        init_dur = init_dur.detach().cpu().numpy()
    noisy_latent_fixed, latent_mask = TTS.sample_noisy_latent(duration=init_dur)
    noisy_latent_fixed = torch.tensor(noisy_latent_fixed, dtype=torch.float32).to(DEVICE) # (1, l_dim, l_len)
    latent_mask = torch.tensor(latent_mask, dtype=torch.float32).to(DEVICE) # (1, 1, l_len)
    
    del tmp_ttl, tmp_dp, TTS


    # ===== 5. Initialize style vectors =====
    if not reference_style:
        print("\nInitializing style randomly (not recommended)")
        _, style_dp = load_voice_style(tmp_voice_path)
        style_ttl = (torch.randn(1, 50, 256) * 0.1)

    elif reference_style == "auto":
        print("\nFinding closest style to target WAV...")

        all_style_paths = sorted(glob.glob("supertonic3/voice_styles/[FM]*.json"))
        best_loss = float('inf') # 1 - similarity
        best_path = None

        for style_path in all_style_paths:
            s_ttl, _ = load_voice_style(style_path) # (1, 50, 256), (1, 8, 16)

            with torch.no_grad():
                _, loss = model(tmp_input_ids, tmp_attention_mask,
                                    s_ttl, vocoder_steps,
                                    noisy_latent_fixed, latent_mask
                                    )

            print(f"  {os.path.basename(style_path)}: {loss:.4f}")
            if loss.item() < best_loss:
                best_loss = loss
                best_path = style_path
        print(f"  >> Best: {os.path.basename(best_path)} (loss={best_loss:.4f})")
        style_ttl, style_dp = load_voice_style(best_path)

        del tmp_input_ids, tmp_attention_mask
    
    elif reference_style:
        print(f"\nInitializing style from: {reference_style}")
        style_ttl, style_dp = load_voice_style(reference_style)

    # set requires_grad
    style_ttl = style_ttl.to(DEVICE).clone().requires_grad_(True) # (1, 50, 256)
    style_dp = style_dp.to(DEVICE).clone() # (1, 8, 16)
    print(f"  style_ttl: {style_ttl.shape}, style_dp: {style_dp.shape} (dp frozen)")


    # ===== Optimization (style_ttl only, style_dp frozen) =====
    max_num_steps = args.num_steps
    save_steps = args.save_steps
    threshold = args.early_stop_loss_threshold
    lr = args.learning_rate
    start_step = 0
    log_step = 8
    best_loss = float('inf')
    best_ttl = None
    best_dp = style_dp.detach().clone()
    
    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=lr * 0.01
    )

    optimizer.zero_grad()
    print(f"\nStarting optimization (step {start_step+1} -> {max_num_steps}, early stop at {threshold})...")
    
    start_time = time.time()
    for step in range(start_step, max_num_steps):
        try:
            # use dataloader
            text_ids_batch, current_text_mask = next(data_iter) # (1, seq_len), (1, 1, seq_len)
        except StopIteration:
            data_iter = iter(dataloader)
            text_ids_batch, current_text_mask = next(data_iter)

        text_ids_batch = text_ids_batch.to(DEVICE)
        current_text_mask = current_text_mask.to(DEVICE)
        # Forward pass
        _, loss = model(
            text_ids_batch, current_text_mask, style_ttl, vocoder_steps, noisy_latent_fixed, latent_mask,
        )

        # Backward + update
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        optimizer.zero_grad()

        step_loss = loss.detach().item() 
        # Track best
        if step_loss < best_loss:
            best_loss = step_loss
            best_ttl = style_ttl.detach().clone()

        # Log
        if (step + 1) % log_step == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step+1}/{max_num_steps} | Loss: {step_loss:.4f} | LR: {current_lr:.4f} | Best: {best_loss:.4f}")

        # Save checkpoint
        if (step + 1) % save_steps == 0:
            ckpt_path = f"{log_dir}/{name}_{step+1:04d}.json"
            save_style(ckpt_path, best_ttl, best_dp, target_wav_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        # Early stopping
        if best_loss <= threshold:
            print(f"  Early stop at step {step+1}: best loss {best_loss:.4f} <= {threshold}")
            break

    # ===== Save final result =====
    final_path = f"{log_dir}/{name}.json"
    print(f"\nSaving best style to: {final_path}")
    save_style(final_path, best_ttl, best_dp, target_wav_path)
    elapsed = time.time() - start_time
    print(f"  Done! Best loss: {best_loss:.4f} | Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
