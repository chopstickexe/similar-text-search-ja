import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from similar_text_search_ja import config
from similar_text_search_ja import es as es_wrapper
from similar_text_search_ja import utils
from similar_text_search_ja.config import Config
from similar_text_search_ja.vectorizers import BaseVectorizer, SentenceVectorizer


def create_index(
    es: "es_wrapper.ES", clear_index: bool, index: str, fields: Dict[str, Any]
):
    """Create ES index

    Args:
        es (es_wrapper.ES): ES wrapper instance
        clear_index (bool): True to delete existing index with the same name
        index (str): index name
        fields (Dict[str, Any]): field mapping settings
    """
    if clear_index:
        es.delete_index(index)

    es.create_index(
        index,
        body={"mappings": {"properties": fields}},
    )


def doc_generator(csv_path: str, batch_size: int):
    file = open(csv_path, newline="", encoding="UTf-8")
    reader = csv.DictReader(file)
    docs = []
    for row in reader:
        docs.append(row)
        if len(docs) == batch_size:
            yield docs
            docs = []
    yield docs


def get_stats(
    docs: List[Dict[str, Any]],
    vectorizer: "BaseVectorizer",
    target_fields: List[str],
):
    logger = logging.getLogger(__name__)
    target_fields = target_fields
    sum_tokens = 0
    sum_chars = 0
    for doc in docs:
        txt = get_target_fields_txt(doc, target_fields)
        sum_chars += len(txt)
        input_ids = vectorizer.encode(txt, padding=False)
        length = input_ids.input_ids.shape[1]
        logger.debug(f"{txt} -> {vectorizer.decode(input_ids)} ({length})")
        sum_tokens += length - 2  # Except cls and sep tokens
    return sum_tokens / len(docs), sum_chars / len(docs)


def vectorize_doc(
    batch_docs: List[Dict[str, Any]],
    target_fields: List[str],
    es_embedding_field: str,
    vectorizer: "BaseVectorizer",
):
    if not batch_docs or len(batch_docs) == 0:
        return
    batch_txts = [get_target_fields_txt(doc, target_fields) for doc in batch_docs]
    outputs = vectorizer.vectorize(batch_txts)
    for i, doc in enumerate(batch_docs):
        doc[es_embedding_field] = outputs[i]


def get_target_fields_txt(doc: Dict[str, Any], target_fields: List[str]) -> str:
    ret = " ".join(
        [doc[field] for field in target_fields if field in doc and doc[field]]
    )
    return ret


def post_documents(
    es: "es_wrapper.ES", index: str, docs: List[Dict[str, Any]], bulk_size: int
):
    """Post documents to ES index

    Args:
        es (es_wrapper.ES): ES wrapper instance
        index (str): Index name
        docs (List[Dict[str, Any]]): Documents
        bulk_size (int): Bulk size
    """
    es.index(index, docs, bulk_size)


def print_summary(report: Path, avg_tokens: float, avg_chars: float):
    with open(str(report), mode="w") as f:
        f.write(f"Average char length = {avg_chars: .3f}\n")
        f.write(f"Average token length = {avg_tokens: .3f}\n")


def __get_es_index_settings(
    base: Dict[str, Any], embedding_field_name: str, embedding_dim: int
):
    base[embedding_field_name] = {
        "type": "dense_vector",
        "dims": embedding_dim,
    }
    return base


def main():
    utils.set_root_logger()
    logger = logging.getLogger(__name__)

    dataset = sys.argv[1]
    conf = Config(dataset)

    config.create_dir(conf.report_dir)

    es = es_wrapper.ES([conf.es_url])

    vectorizer = SentenceVectorizer.create(
        {
            SentenceVectorizer.CONF_KEY_TRAINED_MODEL_PATH: str(conf.model_dir),
            SentenceVectorizer.CONF_KEY_TRANSFORMER_MODEL_NAME: conf.transformer_model,
        }
    )
    field_settings = __get_es_index_settings(
        conf.es_base_index_settings,
        conf.es_embedding_field,
        vectorizer.word_embedding_dim,
    )
    create_index(
        es=es, clear_index=True, index=conf.es_index_name, fields=field_settings
    )

    docs = []
    for batch in doc_generator(conf.raw_csv_path, batch_size=conf.vect_batch_size):
        logger.info(f"Vectorize {len(batch)} documents...")
        vectorize_doc(
            batch,
            conf.target_fields,
            conf.es_embedding_field,
            vectorizer,
        )
        docs.extend(batch)

    avg_tokens, avg_chars = get_stats(docs, vectorizer, conf.target_fields)
    print_summary(conf.report_dir / "index.txt", avg_tokens, avg_chars)

    post_documents(es=es, index=conf.es_index_name, docs=docs, bulk_size=1000)


if __name__ == "__main__":
    main()
