import csv
import logging

import transformers

transformers.BertTokenizer = transformers.BertJapaneseTokenizer

from sentence_transformers import SentenceTransformer, models  # noqa: E402
from sentence_transformers.datasets import SentencesDataset  # noqa: E402
from sentence_transformers.evaluation import TripletEvaluator  # noqa: E402
from sentence_transformers.losses import (  # noqa: E402
    TripletDistanceMetric,
    TripletLoss,
)
from sentence_transformers.readers import TripletReader  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# Reference
# https://github.com/UKPLab/sentence-transformers/blob/v0.3.9/examples/training/other/training_wikipedia_sections.py

BATCH_SIZE = 8
NUM_EPOCHS = 4
EVAL_STEPS = 200
DATASET_NAME = "20200919"
OUTPUT_PATH = "models/mlit/" + DATASET_NAME
DATA_PATH = "data/mlit/triplets/" + DATASET_NAME

transformer = models.BERT("cl-tohoku/bert-base-japanese-whole-word-masking")

pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[transformer, pooling], device="cuda")

triplet_reader = TripletReader(DATA_PATH)
train_data = SentencesDataset(triplet_reader.get_examples("train.tsv"), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
train_loss = TripletLoss(
    model=model, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=1
)

anchors = []
positives = []
negatives = []
with open(DATA_PATH + "/dev.tsv", mode="r", encoding="UTF-8", newline="") as f:
    for row in csv.reader(f, delimiter="\t"):
        anchors.append(row[0])
        positives.append(row[1])
        negatives.append(row[2])

dev_data = SentencesDataset(triplet_reader.get_examples("dev.tsv"), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=BATCH_SIZE)
evaluator = TripletEvaluator(anchors, positives, negatives, name="dev")

logging.info("Read Wikipedia Triplet dev dataset")
dev_examples = []


WARMUP_STEPS = int(len(train_data) // BATCH_SIZE * 0.1)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=NUM_EPOCHS,
    evaluation_steps=EVAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    output_path=OUTPUT_PATH,
)

logging.info("Read test examples")
anchors = []
positives = []
negatives = []
with open(DATA_PATH + "/test.tsv", mode="r", encoding="UTF-8", newline="") as f:
    for row in csv.reader(f, delimiter="\t"):
        anchors.append(row[0])
        positives.append(row[1])
        negatives.append(row[2])


model = SentenceTransformer(OUTPUT_PATH, device="cuda")
test_evaluator = TripletEvaluator(anchors, positives, negatives, name="test")
test_evaluator(model, output_path=OUTPUT_PATH)