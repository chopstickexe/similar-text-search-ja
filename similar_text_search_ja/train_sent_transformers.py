import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.datasets import SentencesDataset
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
from sentence_transformers.readers import TripletReader
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from similar_text_search_ja import config, utils
from similar_text_search_ja.config import Config

# Reference
# https://github.com/UKPLab/sentence-transformers/blob/v0.3.9/examples/training/other/training_wikipedia_sections.py


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    eval_steps: int
    model_dir: Path
    transformer_model: str
    train_triplets_tsv: Path
    dev_triplets_tsv: Path
    test_triplets_tsv: Path


def __get_triplet_evaluator(tsv_path: Path, name: str):
    anchors = []
    positives = []
    negatives = []
    with open(str(tsv_path), mode="r", encoding="UTF-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            anchors.append(row[0])
            positives.append(row[1])
            negatives.append(row[2])
    return TripletEvaluator(anchors, positives, negatives, name=name)


def train(conf: "TrainConfig"):
    logger = logging.getLogger(__name__)
    logger.info("Initialize model")
    transformer = models.Transformer(conf.transformer_model)

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[transformer, pooling])
    model.tokenizer = AutoTokenizer.from_pretrained(conf.transformer_model)
    logger.info(f"model: {type(model)}")
    logger.info(f"tokenizer: {type(model.tokenizer)}")
    encode_result = model.tokenizer(["日本語のトークナイゼーションの問題"], return_tensors='pt', padding=True)
    logger.info(model.tokenizer.convert_ids_to_tokens(encode_result.input_ids.flatten().tolist()))

    logger.info("Read training data")
    triplet_reader = TripletReader(str(conf.train_triplets_tsv.parent))
    train_data = SentencesDataset(
        triplet_reader.get_examples(conf.train_triplets_tsv.name), model=model
    )
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=conf.batch_size)
    train_loss = TripletLoss(
        model=model, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=1
    )

    evaluator = TripletEvaluator.from_input_examples(
        triplet_reader.get_examples(conf.dev_triplets_tsv.name), name="dev"
    )

    logger.info("Start training")
    warmup_steps = int(len(train_data) // conf.batch_size * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=conf.epochs,
        evaluation_steps=conf.eval_steps,
        warmup_steps=warmup_steps,
        output_path=str(conf.model_dir),
    )


def test(conf: "TrainConfig"):
    logger = logging.getLogger(__name__)
    logger.info("Start test")

    model = SentenceTransformer(str(conf.model_dir))
    model.tokenizer = AutoTokenizer.from_pretrained(conf.transformer_model)
    logger.info(f"model: {type(model)}")
    logger.info(f"tokenizer: {type(model.tokenizer)}")

    encode_result = model.tokenizer(["日本語のトークナイゼーションの問題"], return_tensors='pt', padding=True)
    logger.info(model.tokenizer.convert_ids_to_tokens(encode_result.input_ids.flatten().tolist()))

    triplet_reader = TripletReader(str(conf.train_triplets_tsv.parent))
    evaluator = TripletEvaluator.from_input_examples(
        triplet_reader.get_examples(conf.test_triplets_tsv.name), name="test"
    )
    evaluator(model, output_path=str(conf.model_dir))


def init(dataset: str):
    conf = Config(dataset)
    model_dir = conf.model_dir
    config.create_dir(model_dir)

    train_conf = TrainConfig(
        conf.sent_transformer_train_batch_size,
        conf.sent_transformer_train_epochs,
        conf.sent_transformer_train_eval_steps,
        conf.model_dir,
        conf.transformer_model,
        conf.data_triplets_train_path,
        conf.data_triplets_dev_path,
        conf.data_triplets_test_path
    )

    return train_conf


def main():
    utils.set_root_logger()

    dataset = sys.argv[1]
    conf = init(dataset)
    train(conf)
    test(conf)


if __name__ == "__main__":
    main()
