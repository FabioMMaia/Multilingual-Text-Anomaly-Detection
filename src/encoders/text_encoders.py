import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

class BaseTextEncoder(ABC):
    @abstractmethod
    def encode_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        pass

class SentenceBERT(BaseTextEncoder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)

class BERTimbau(BaseTextEncoder):
    def __init__(self, model_name: str, use_cls=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_cls = use_cls

    def encode_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state

                if self.use_cls:
                    embeddings = last_hidden[:, 0, :].cpu().numpy()
                else:
                    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                    summed = torch.sum(last_hidden * mask, dim=1)
                    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    embeddings = (summed / summed_mask).cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

