import logging
from pathlib import Path
from typing import Any, Dict, List

from similar_text_search_ja import es as es_wrapper
from similar_text_search_ja import index as index_base
from similar_text_search_ja import utils
from similar_text_search_ja.mlit import config
from similar_text_search_ja.vectorizers import (BaseVectorizer,
                                                HuggingfaceVectorizer)


def __create_index(es: "es_wrapper.ES", conf: Dict[str, Any]):
    mlit_conf = conf["mlit"]
    index_base.create_index(
        es,
        mlit_conf["es_always_clear_index"],
        mlit_conf["es_index"],
        {
            mlit_conf["date_field"]: {
                "type": "date",
                "format": mlit_conf["date_format"],
            },
            conf["bert_cls_field"]: {
                "type": "dense_vector",
                "dims": conf["bert_vec_size"],
            },
        },
    )


def __get_documents(vectorizer: "BaseVectorizer", conf: Dict[str, Any]):
    logger = logging.getLogger(__name__)
    mlit_conf = conf["mlit"]

    docs = []
    for batch in index_base.doc_generator(
        mlit_conf["csv_path"], mlit_conf["batch_size"]
    ):
        index_base.add_vectors(
            batch,
            mlit_conf["target_fields"],
            conf["bert_cls_field"],
            vectorizer,
        )
        docs.extend(batch)
        if len(docs) % 1000 == 0:
            logger.info(f"Vectorized {len(docs)} documents...")
    return docs


def __post_documents(
    es: "es_wrapper.ES", docs: List[Dict[str, Any]], conf: Dict[str, Any]
):
    mlit_conf = conf["mlit"]
    index_base.post_documents(
        es, mlit_conf["es_index"], docs, mlit_conf["es_bulk_size"]
    )


def main():
    utils.set_root_logger()

    conf = config.get_config()

    report_path = Path(conf["mlit"]["index_report_path"])
    index_base.create_report_dir(report_path)

    es = es_wrapper.ES([conf["es_url"]])

    __create_index(es, conf)

    vectorizer_config = {
        "tokenizer_name_or_path": "cl-tohoku/bert-base-japanese-whole-word-masking",
        "model_name_or_path": "cl-tohoku/bert-base-japanese-whole-word-masking",
    }
    vectorizer = HuggingfaceVectorizer.create(vectorizer_config)
    docs = __get_documents(vectorizer, conf)
    avg_tokens, avg_chars = index_base.get_stats(docs, vectorizer, conf)
    index_base.print_summary(report_path, avg_tokens, avg_chars)

    __post_documents(es, docs, conf)


if __name__ == "__main__":
    main()
