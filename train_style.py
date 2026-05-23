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

import os
import glob
import time
import numpy as np
import torch

from helper import load_text_to_speech, load_voice_style as load_voice_styles
from utils import save_style, SupertonicModel, get_train_dataloader
from configs import texts, TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_voice_style(path):
    """ load single voice style """
    style = load_voice_styles([path]) # Style
    style_ttl = torch.tensor(style.ttl, dtype=torch.float32).to(DEVICE) # (1, 50, 256)
    style_dp = torch.tensor(style.dp, dtype=torch.float32).to(DEVICE) # (1, 8, 16)
    return style_ttl, style_dp


def main():
    # ===== 0. configs =====
    config = TrainConfig()
    name= config.NAME
    gender= config.GENDER
    target_wav_path= config.TARGET_WAV_PATH
    target_lang= config.TARGET_WAV_LANG
    reference_style= config.REFERENCE_STYLE
    seed= config.SEED
    speed= config.SPEED
    ve_steps = config.VE_STEPS
    

    # models
    model = SupertonicModel("supertonic3/onnx", target_wav_path)
    TTS = load_text_to_speech("supertonic3/onnx")

    print(f"Using device: {DEVICE}")
    print(f"Name: {name}")
    
    # Paths
    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # ===== 4. Generate fixed noisy latent (seed-controlled) =====
    # create dataset
    train_texts = [ t for t in texts if t[0]==target_lang]
    dataloader = get_train_dataloader(TTS, train_texts)
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
                                    s_ttl, ve_steps,
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


    # ===== 6. Optimization (style_ttl only, style_dp frozen) =====
    max_num_steps= config.NUM_STEPS
    save_steps= config.SAVE_STEPS
    threshold= config.EARLY_STOP_LOSS_THRESHOLD
    lr = config.LEARNING_RATE
    start_step = 0
    log_step = 20
    lr_update_steps = max_num_steps // 20
    best_loss = float('inf')
    best_ttl = None
    best_dp = style_dp.detach().clone()
    
    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=lr_update_steps
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
            text_ids_batch, current_text_mask, style_ttl, ve_steps, noisy_latent_fixed, latent_mask,
        )

        # Backward + update
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        if (step + 1) % lr_update_steps == 0:
            scheduler.step()

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

    # ===== 7. Save final result =====
    final_path = f"{log_dir}/{name}.json"
    print(f"\nSaving best style to: {final_path}")
    save_style(final_path, best_ttl, best_dp, target_wav_path)
    elapsed = time.time() - start_time
    print(f"  Done! Best loss: {best_loss:.4f} | Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
