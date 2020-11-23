from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class BaseVectorizer(ABC):
    @classmethod
    @abstractmethod
    def create(cls: "BaseVectorizer", config: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def encode(self, sentences: List[str], padding: bool = True):
        raise NotImplementedError

    @abstractmethod
    def decode(self, encode_result):
        raise NotImplementedError

    @abstractmethod
    def vectorize(self, input_ids):
        raise NotImplementedError


class HuggingfaceVectorizer(BaseVectorizer):
    """Vectorize text by using a Huggingface transformer"""

    CONF_KEY_MODEL_NAME = "model_name"

    def __init__(self, model_name: str):
        # Reference:
        # https://github.com/UKPLab/sentence-transformers/blob/e0aa596a0397a41ba69f75c1124318f0cb1dceca/sentence_transformers/models/Transformer.py
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            return_dict=True,
        )
        self.word_embedding_dim = self.model.config.hidden_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    @classmethod
    def create(cls: "HuggingfaceVectorizer", config: Dict[str, Any]):
        return cls(config[cls.CONF_KEY_MODEL_NAME])

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
    """Vectorize text by using sentence transformers
    https://github.com/UKPLab/sentence-transformers
    """

    CONF_KEY_TRAINED_MODEL_PATH = "model_path"
    CONF_KEY_TRANSFORMER_MODEL_NAME = "transformer_model_name"

    def __init__(self, model_path: str, transformer_model_name: str):
        # Reference:
        # https://github.com/UKPLab/sentence-transformers/blob/e0aa596a0397a41ba69f75c1124318f0cb1dceca/sentence_transformers/models/Transformer.py
        self.model = SentenceTransformer(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model.tokenizer = self.tokenizer
        self.word_embedding_dim = self.model.get_sentence_embedding_dimension()

    @classmethod
    def create(cls: "SentenceVectorizer", config: Dict[str, Any]):
        return cls(
            config[cls.CONF_KEY_TRAINED_MODEL_PATH],
            config[cls.CONF_KEY_TRANSFORMER_MODEL_NAME],
        )

    def encode(self, sentences: List[str], padding: bool = True):
        if len(sentences) == 0:
            return None
        return self.model.tokenizer(sentences, return_tensors="pt", padding=padding)

    def decode(self, encode_result):
        return self.model.tokenizer.convert_ids_to_tokens(
            encode_result.input_ids.flatten().tolist()
        )

    def vectorize(self, sentences):
        return self.model.encode(sentences)
