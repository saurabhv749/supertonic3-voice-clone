import torch
from torch.utils.data import TensorDataset, DataLoader


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_train_dataloader(tts, texts, batch_size=1):
    "Returns tokenized text dataloader"

    def preprocess_text(x):
        return len(tts.text_processor._preprocess_text(text=x[1], lang=x[0]))

    # Texts for multi-text rotation
    sorted_texts = sorted(texts, reverse=True, key= preprocess_text)
    text_languages = [t[0] for t in sorted_texts]
    train_texts = [t[1] for t in sorted_texts]

    ids_np, mask_np = tts.text_processor(train_texts, text_languages) # ((B,seq_len), (B, 1, max_len))
    input_ids = torch.tensor(ids_np, dtype=torch.long).to(DEVICE) # (B,seq_len)
    attention_mask = torch.tensor(mask_np, dtype=torch.long).to(DEVICE) # (B, 1, max_len)

    # create dataset
    train_ds = TensorDataset(input_ids, attention_mask)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    return train_dataloader
