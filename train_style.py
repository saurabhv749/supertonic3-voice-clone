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

import json
import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
import onnxslim
import onnx
from onnx import shape_inference
import onnx2torch
from onnx2torch import convert

from helper import load_text_to_speech, load_voice_style as load_voice_styles
from utils import (load_speaker_model, 
                   embed_audio, speaker_similarity_loss, 
                   load_audio_16khz_mono,load_texts_and_languages
                   )

# SSL certificate workaround
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
import httpx
_orig_client = httpx.Client
class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = _NoVerifyClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== ONNX to PyTorch conversion =====

def _patch_onnx2torch():
    """Bypass onnx2torch's safe_shape_inference which writes temp files."""
    def patched(m):
        if isinstance(m, str):
            m = onnx.load(m)
        try:
            return shape_inference.infer_shapes(m)
        except:
            return m
    onnx2torch.converter.safe_shape_inference = patched

def _fix_clip(model):
    """Remove empty Clip inputs that cause onnx2torch conversion errors."""
    # required for converting: duration_predictor.onnx and text_encoder.onnx
    for node in model.graph.node:
        if node.op_type == 'Clip':
            inputs = list(node.input)
            while inputs and inputs[-1] == '':
                inputs.pop()
            del node.input[:]
            node.input.extend(inputs)
    return model

def load_pt_model(name, onnx_dir="supertonic3/onnx"):
    """Load ONNX model, slim it, fix opset, and convert to PyTorch."""
    slimmed = onnxslim.slim(os.path.join(onnx_dir, name))
    for opset in slimmed.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            opset.version = 17
    _fix_clip(slimmed)
    m = convert(slimmed)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEVICE)


# ===== TTS forward pass =====

def tts_forward(text_ids, text_mask, style_ttl, style_dp,
                dp_model, te_model, ve_model, voc_model,
                total_step, speed, noisy_latent, latent_mask):
    """Differentiable TTS forward pass through all 4 models."""
    # text_ids: (B, seq_len) int64
    # text_mask: (B, 1, seq_len) float
    
    # style_dp: (B, 8, 16) float
    # style_ttl: (B, 50, 256) float

    # noisy_latent: (B, 144 ,seq_len) float
    # latent_mask: (B, 1, seq_len) float
    dur = dp_model(text_ids, style_dp, text_mask) 
    dur = dur / speed
    text_emb = te_model(text_ids, style_ttl, text_mask)
    xt = noisy_latent * latent_mask
    total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
    for step in range(total_step):
        current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
        xt = ve_model(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)
    wav = voc_model(xt)
    return wav, dur

# ===== Save style JSON =====

def save_style(path, style_ttl, style_dp, source_file=None):
    """Save style vectors in SupertonicTTS-compatible JSON format."""
    from datetime import datetime
    style_json = {
        "style_ttl": {
            "data": style_ttl.detach().cpu().numpy().tolist(),
            "dims": [1, 50, 256],
            "type": "float32"
        },
        "style_dp": {
            "data": style_dp.detach().cpu().numpy().tolist(),
            "dims": [1, 8, 16],
            "type": "float32"
        },
        "metadata": {
            "source_file": source_file or "unknown",
            "source_sample_rate": 44100,
            "target_sample_rate": 44100,
            "extracted_at": datetime.now().isoformat()
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(style_json, f)

# ===== Main =====

def main():
    _patch_onnx2torch()

    # ===== 0. configs =====

    arg = sys.argv[1] if len(sys.argv) > 1 else "default"
    if os.path.exists(arg):
        config_path = arg
    elif os.path.exists(f"configs/{arg}.json"):
        config_path = f"configs/{arg}.json"
    elif os.path.exists(f"configs/{arg}"):
        config_path = f"configs/{arg}"
    else:
        print(f"Config not found: {arg}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name = cfg["name"]
    gender = cfg["gender"]
    target_wav_path = cfg["target_wav"]
    reference_style = cfg.get("reference_style")
    seed = cfg.get("seed", 42)
    lr = cfg.get("lr", 2e-4)
    num_steps = cfg.get("num_steps", 3000)
    total_step = cfg.get("total_step", 5)
    speed = cfg.get("speed", 1.05)
    save_every = cfg.get("save_every", 100)
    threshold = cfg.get("early_stop_loss_threshold", 0.16)


    # Paths
    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)

    # Save config to log dir
    with open(os.path.join(log_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


    print(f"Using device: {DEVICE}")
    print(f"Name: {name}")

    # ===== 1. Load target WAV and extract features =====
    print(f"\nLoading target WAV: {target_wav_path}")
    target_wav_t = load_audio_16khz_mono(target_wav_path).to(DEVICE) # (1,len)
    print(f"  Duration: {target_wav_t.shape[-1]/16000:.2f}s")

    print("\nLoading voice embedding model...")
    voice_encoder = load_speaker_model()
    print("\nvoice embedding model loaded.")

    print("Extracting target features...")
    target_feats = embed_audio(voice_encoder, target_wav_path) # ( 192,)
    print("  Done.")

    # ===== 2. Load TTS models (ONNX -> PyTorch) =====
    print("\nConverting ONNX models to PyTorch...")
    dp_model = load_pt_model("duration_predictor.onnx")
    te_model = load_pt_model("text_encoder.onnx")
    ve_model = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")
    print("  All models converted.")

    # ===== 3. Preprocess texts =====
    tts = load_text_to_speech("supertonic3/onnx")
    # Texts for multi-text rotation
    train_texts, text_languages = load_texts_and_languages('configs/utterances.txt')

    ids_np, mask_np = tts.text_processor(train_texts, text_languages) # ((B,seq_len), (B, 1, max_len))
    input_ids = torch.tensor(ids_np, dtype=torch.long).to(DEVICE) # (B,seq_len)
    attention_mask = torch.tensor(mask_np, dtype=torch.long).to(DEVICE) # (B, 1, max_len)

    # create dataset
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)
    # ===== 4. Generate fixed noisy latent (seed-controlled) =====
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use the first text as a reference for latent length and style extraction
    tmp_input_ids = input_ids[0].unsqueeze(0) # 1, seq_len
    tmp_attention_mask = attention_mask[0].unsqueeze(0) # 1, 1, seq_len
    # v3
    def load_voice_style(path):
        """ load single voice style """
        style = load_voice_styles([path]) # Style
        style_ttl = torch.tensor(style.ttl, dtype=torch.float32).to(DEVICE) # (1, 50, 256)
        style_dp = torch.tensor(style.dp, dtype=torch.float32).to(DEVICE) # (1, 8, 16)
        return style_ttl, style_dp


    tmp_voice = "F4.json" if gender == "F" else "M1.json"
    tmp_voice_path = f"supertonic3/voice_styles/{tmp_voice}"
    tmp_ttl, tmp_dp = load_voice_style(tmp_voice_path) # (1, 50, 256), (1, 8, 16)

    # latent computation
    with torch.no_grad():
        init_dur = dp_model(tmp_input_ids, tmp_dp, tmp_attention_mask) / speed
    noisy_latent_fixed, latent_mask = tts.sample_noisy_latent(duration=init_dur.detach().cpu().numpy())
    noisy_latent_fixed = torch.tensor(noisy_latent_fixed, dtype=torch.float32).to(DEVICE) # (1, l_dim, l_len)
    latent_mask = torch.tensor(latent_mask, dtype=torch.float32).to(DEVICE) # (1, 1, l_len)
    print(f"Noisy latent: {noisy_latent_fixed.shape}, mask: {latent_mask.shape}")
    
    del tmp_ttl, tmp_dp, tts


    # ===== 5. Initialize style vectors =====
    if not reference_style:
        print("\nInitializing style randomly (not recommended)")
        _, style_dp = load_voice_style(tmp_voice_path)
        style_ttl = (torch.randn(1, 50, 256) * 0.1)

    elif reference_style == "auto":
        print("\nFinding closest style to target WAV (WavLM Layer 3)...")
        
        all_style_paths = sorted(glob.glob("supertonic3/voice_styles/[FM]*.json"))
        best_dist = float('inf') # 1 - similarity
        best_path = None

        for style_path in all_style_paths:
            s_ttl, s_dp = load_voice_style(style_path) # (1, 50, 256), (1, 8, 16)

            with torch.no_grad():
                test_wav, _ = tts_forward(
                    tmp_input_ids, tmp_attention_mask,
                    s_ttl, s_dp,
                    dp_model, te_model, ve_model, voc_model,
                    total_step, speed, noisy_latent_fixed, latent_mask,
                )
                dist = speaker_similarity_loss(voice_encoder, target_feats, test_wav)
            print(f"  {os.path.basename(style_path)}: {dist:.4f}")
            if dist < best_dist:
                best_dist = dist
                best_path = style_path
        print(f"  >> Best: {os.path.basename(best_path)} (loss={best_dist:.4f})")
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
    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=lr * 0.01
    )


    start_step = 0
    log_step = 8
    end_step = num_steps
    start_time = time.time()
    print(f"\nStarting optimization (step {start_step+1} -> {end_step}, early stop at {threshold})...")

    best_loss = float('inf')
    best_ttl = None
    best_dp = style_dp.detach().clone()
    
    optimizer.zero_grad()
    for step in range(start_step, end_step):
        try:
            # use dataloader
            text_ids_batch, current_text_mask = next(data_iter) # (1, seq_len), (1, 1, seq_len)
        except StopIteration:
            data_iter = iter(dataloader)
            text_ids_batch, current_text_mask = next(data_iter)

        text_ids_batch = text_ids_batch.to(DEVICE)
        current_text_mask = current_text_mask.to(DEVICE)
        # Forward pass
        wav_out, _ = tts_forward(
            text_ids_batch, current_text_mask, style_ttl, style_dp,
            dp_model, te_model, ve_model, voc_model,
            total_step, speed, noisy_latent_fixed, latent_mask,
        )

        # Compute loss
        loss = speaker_similarity_loss(voice_encoder, target_feats, wav_out)
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
            print(f"  Step {step+1}/{end_step} | Loss: {step_loss:.4f} | LR: {current_lr:.4f} | Best: {best_loss:.4f}")

        # Save checkpoint
        if (step + 1) % save_every == 0:
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
