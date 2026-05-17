import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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

class SpeakerID(torch.nn.Module):
    
    def __init__(self, target_wav_path: str):
        super().__init__()
        self.encoder = self.get_speaker_encoder()
        
        self.target_wav_path = target_wav_path
        self.emb_original = self.embed_audio(target_wav_path, verbose=True) # ( 192,)
        self.norm_emb_original = torch.nn.functional.normalize(self.emb_original, p=2, dim=0)

    def get_speaker_encoder(self):
        """Loads the speaker verification model (ECAPA-VOXCELEB).
        
        Returns:
            EncoderClassifier: Pre-trained speaker recognition model on GPU/CPU.
                            Model is set to eval mode with gradients disabled.
        """
        
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={'device': DEVICE}
        )
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)
        return classifier

    def embed_wav(self, wav):
        """Extracts speaker embeddings from an audio waveform.
        
        Args:
            wav (torch.Tensor): Audio waveform with shape (1, T), where T is the number of samples.
        
        Returns:
            torch.Tensor: Speaker embedding with shape (embd_dim,).
        """
        embeddings = self.encoder.encode_batch(wav)
        return embeddings.squeeze(0).squeeze(0) # (dim)

    def embed_audio(self, audio_path, verbose=False):
        """Loads an audio file and extracts its speaker embeddings.
        
        Args:
            audio_path (str): Path to the audio file to embed.
        
        Returns:
            torch.Tensor: Speaker embedding with shape (embd_dim,).
        """
        wav = load_audio_16khz_mono(audio_path) # (1, T)
        if verbose:
            print(f"Loaded: {audio_path} | Duration: {wav.shape[-1]/16000:.2f}s")
        # The encode_batch method expects a batch, so we expand dimensions
        embedding = self.embed_wav(wav)
        embedding = embedding.to(DEVICE)
        return embedding

    def speaker_similarity_loss(self, generated_wav):
        """Calculates the speaker similarity loss between original and generated audio.
        
        Args:
            generated_wav (torch.Tensor): Generated audio waveform with shape (1, T) at 44100 Hz.
        
        Returns:
            torch.Tensor: Similarity loss (scalar), computed as 1 - cosine_similarity.
                        Lower values indicate higher speaker similarity.
        """
        # convert generated_wav to 16Khz
        signal = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(generated_wav.cpu())
        emb_generated = self.embed_wav(signal)
        # Normalize embeddings
        norm_emb_generated = torch.nn.functional.normalize(emb_generated, p=2, dim=0)
        
        cosine_sim = torch.dot(self.norm_emb_original, norm_emb_generated)
        loss = 1 - cosine_sim
        return loss
