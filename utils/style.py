import os
import json

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
