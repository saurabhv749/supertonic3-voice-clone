import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# AUDIO & SPEAKER EMBEDDING UTILS
def load_audio_16khz_mono(file_path):
    """Loads an audio file and resamples to 16kHz mono.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        torch.Tensor: Audio signal with shape (1, T), where T is the number of samples.
    """
    signal, _ = torchaudio.load(file_path)
    # Resample to 16kHz if necessary
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)  # Convert to mono
    if _ != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=_, new_freq=16000)(signal)
    return signal

def load_speaker_model():
    """Loads the speaker verification model (ECAPA-VOXCELEB).
    
    Returns:
        EncoderClassifier: Pre-trained speaker recognition model on GPU/CPU.
                          Model is set to eval mode with gradients disabled.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
    run_opts={'device':device}
    )
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)
    return classifier

def embed_wav(model, wav):
    """Extracts speaker embeddings from an audio waveform.
    
    Args:
        model (EncoderClassifier): Pre-trained speaker verification model.
        wav (torch.Tensor): Audio waveform with shape (1, T), where T is the number of samples.
    
    Returns:
        torch.Tensor: Speaker embedding with shape (embd_dim,).
    """
    embeddings = model.encode_batch(wav)
    return embeddings.squeeze(0).squeeze(0) # (dim)

def embed_audio(model, audio_path):
    """Loads an audio file and extracts its speaker embeddings.
    
    Args:
        model (EncoderClassifier): Pre-trained speaker verification model.
        audio_path (str): Path to the audio file to embed.
    
    Returns:
        torch.Tensor: Speaker embedding with shape (embd_dim,).
    """
    wav = load_audio_16khz_mono(audio_path) # (1, T)
    # The encode_batch method expects a batch, so we expand dimensions
    embedding = embed_wav(model, wav)
    return embedding

def speaker_similarity_loss(model, original_embd, generated_wav):
    """Calculates the speaker similarity loss between original and generated audio.
    
    Args:
        model (EncoderClassifier): Pre-trained speaker verification model.
        original_embd (torch.Tensor): Speaker embedding from original audio with shape (embd_dim,).
        generated_wav (torch.Tensor): Generated audio waveform with shape (1, T) at 44100 Hz.
    
    Returns:
        torch.Tensor: Similarity loss (scalar), computed as 1 - cosine_similarity.
                     Lower values indicate higher speaker similarity.
    """
    # convert generated_wav to 16Khz
    signal = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(generated_wav.cpu())
    emb_generated = embed_wav(model, signal)
    # Normalize embeddings
    norm_emb_original = torch.nn.functional.normalize(original_embd, p=2, dim=0)
    norm_emb_generated = torch.nn.functional.normalize(emb_generated, p=2, dim=0)

    cosine_sim = torch.dot(norm_emb_original, norm_emb_generated)

    loss = 1 - cosine_sim
    return loss

# TEXT PREPROCESSING UTILS
def load_texts_and_languages(file_path):
    """Loads texts and their corresponding languages from a file.
    
    Args:
        file_path (str): Path to the text file containing lines in the format "language|text".
    
    Returns:
        tuple: A tuple containing two lists:
            - texts (list of str): List of text strings.
            - languages (list of str): List of corresponding language codes.
    """
    texts, languages = [], []
    with open(file_path, 'r') as f:
        for line in f:
            lang, text = line.split('|', 1)
            languages.append(lang.strip())
            texts.append(text.strip())
    return texts, languages