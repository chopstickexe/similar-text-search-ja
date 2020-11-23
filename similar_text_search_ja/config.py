from pathlib import Path

from similar_text_search_ja import utils


class Config:
    def __init__(self, dataset: str):
        config_file = utils.get_dir().parent / "config.json"
        self._values = utils.read_json_config(config_file)
        self.es_url = self._values["es_url"]
        self.es_embedding_field = self._values["es_embedding_field"]
        self.transformer_model = self._values["transformer_model"]

        self._dataset = self._values["datasets"][dataset]
        self.es_always_clear_index = self._dataset["es_always_clear_index"]
        self.es_bulk_size = self._dataset["es_bulk_size"]
        self.es_base_index_settings = self._dataset["es_base_index_settings"]
        self.vect_batch_size = self._dataset["vect_batch_size"]
        self.raw_csv_path = Path(self._dataset["raw_csv_path"])
        self.target_fields = self._dataset["target_fields"]
        self.ans_field = self._dataset["ans_field"]
        self.invalid_ans = self._dataset["invalid_ans"]

        self.es_index_name = dataset

        self.data_dir = Path("data") / dataset
        self.data_triplets_dir = self.data_dir / "triplets"
        self.data_triplets_train_path = self.data_triplets_dir / "train.tsv"
        self.data_triplets_dev_path = self.data_triplets_dir / "dev.tsv"
        self.data_triplets_test_path = self.data_triplets_dir / "test.tsv"

        self.model_dir = Path("models") / dataset
        self.report_dir = Path("reports") / dataset


def create_dir(dir: Path):
    dir.mkdir(parents=True, exist_ok=True)
