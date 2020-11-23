from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoTokenizer


class BaseVectorizer(ABC):
    @classmethod
    @abstractmethod
    def create(cls: "BaseVectorizer", config: Dict[str, Any]):
        raise NotImplementedError


@dataclass
class HuggingfaceVectorizerConfig:
    tokenizer_name_or_path: str
    model_name_or_path: str


class HuggingfaceVectorizer(BaseVectorizer):
    """Vectorize text by using a Huggingface transformer"""

    def __init__(self, tokenizer_name_or_path: str, model_name_or_path):
        # Reference:
        # https://github.com/UKPLab/sentence-transformers/blob/e0aa596a0397a41ba69f75c1124318f0cb1dceca/sentence_transformers/models/Transformer.py
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.model = AutoModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            output_hidden_states=True,
            return_dict=True,
        )
        self.word_embedding_dim = self.model.config.hidden_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    @classmethod
    def create(cls: "BaseVectorizer", config: Dict[str, Any]):
        return cls(config["tokenizer_name_or_path"], config["model_name_or_path"])

    def encode(self, sentences: List[str], padding: bool = True):
        if len(sentences) == 0:
            return None
        return self.tokenizer(sentences, return_tensors="pt", padding=padding)

    def decode(self, encode_result):
        return self.tokenizer.convert_ids_to_tokens(
            encode_result.input_ids.flatten().tolist()
        )

    def vectorize(self, input_ids):
        input_ids.to(self.device)
        with torch.no_grad():
            return self.model(**input_ids)


class SentenceVectorizer(BaseVectorizer):
    pass
