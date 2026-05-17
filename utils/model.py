import os
import torch
import torch.nn as nn
import onnx
from onnx import shape_inference
import onnxslim
import onnx2torch
from onnx2torch import convert
import httpx

from .loss import SpeakerID

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# SSL certificate workaround
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
_orig_client = httpx.Client

class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = _NoVerifyClient

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

def load_pt_model(onnx_dir, name):
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


_patch_onnx2torch()


class SupertonicModel(nn.Module):
    def __init__(self, onnx_dir:str, target_wav_path: str):
        super().__init__()
        self.voice_encoder = SpeakerID(target_wav_path=target_wav_path)
        self.load_models(onnx_dir)
    
    def load_models(self, onnx_dir:str):
        print("\n[utils.model]: Converting ONNX models to PyTorch...")
        self.dp_model = load_pt_model(onnx_dir, "duration_predictor.onnx")
        self.te_model = load_pt_model(onnx_dir, "text_encoder.onnx")
        self.ve_model = load_pt_model(onnx_dir, "vector_estimator.onnx")
        self.voc_model = load_pt_model(onnx_dir, "vocoder.onnx")
        print("[utils.model]: ✅ All models converted to PyTorch.")
    
    def forward(self, text_ids, text_mask, style_ttl,
                total_step, noisy_latent, latent_mask,
                ):
        """
        Differentiable TTS forward pass through all 4 models.
        INPUTS:
            text_ids: (B, seq_len) int64
            text_mask: (B, 1, seq_len) float
            style_ttl: (B, 50, 256) float
            total_step: float
            noisy_latent: (B, 144 ,seq_len) float
            latent_mask: (B, 1, seq_len) float
        """
        text_emb = self.te_model(text_ids, style_ttl, text_mask)
        xt = noisy_latent * latent_mask
        total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
        for step in range(total_step):
            current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
            xt = self.ve_model(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)
        wav = self.voc_model(xt)
        loss = self.voice_encoder.speaker_similarity_loss(wav)
        return wav, loss
