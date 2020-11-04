import torch

from transformers.modeling_bert import BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from typing import List


class JaVectorizer:
    def __init__(self):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.model = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            output_hidden_states=True,
            return_dict=True,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def encode(self, sentences: List[str]):
        if len(sentences) == 0:
            return None
        input_ids = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = input_ids.to(self.device)
        return input_ids

    def vectorize(self, input_ids):
        with torch.no_grad():
            return self.model(**input_ids)