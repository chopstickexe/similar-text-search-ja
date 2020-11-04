import es as es_wrapper
import utils
from csv_parser import CsvParser
from vectorizer import JaVectorizer
from typing import Dict, Any, List
import logging

def create_index(
    es: "es_wrapper.ES",
    clear_index: bool,
    index: str,
    date_field: str,
    date_format: str,
    bert_cls_field: str,
    bert_vec_size: int,
):
    if clear_index:
        es.delete_index(index)

    es.create_index(
        index,
        body={
            "mappings": {
                "properties": {
                    date_field: {"type": "date", "format": date_format},
                    bert_cls_field: {"type": "dense_vector", "dims": bert_vec_size},
                }
            }
        },
    )


def get_target_fields_txt(doc: Dict[str, Any], target_fields: List[str]) -> str:
    ret = ' '.join([doc[field] for field in target_fields if field in doc and doc[field]])
    return ret


def vectorize_doc(batch_docs, target_fields, bert_cls_field, vectorizer):
    if not batch_docs or len(batch_docs) == 0:
        return
    batch_txts = [get_target_fields_txt(doc, target_fields) for doc in batch_docs]
    input_ids = vectorizer.encode(batch_txts)
    outputs = vectorizer.vectorize(input_ids)
    assert len(outputs.last_hidden_state) == len(batch_docs)
    for i, doc in enumerate(batch_docs):
        doc[bert_cls_field] = outputs.last_hidden_state[i][0][:].tolist()


def get_documents(
    csv_path: str,
    target_fields: List[str],
    bert_cls_field: str,
    vectorizer: "JaVectorizer",
) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    ret = []
    with CsvParser(csv_path) as csv_parser:
        batch_docs = []
        for doc in csv_parser:
            batch_docs.append(doc)
            if len(batch_docs) == 32:
                vectorize_doc(batch_docs, target_fields, bert_cls_field, vectorizer)
                ret.extend(batch_docs)
                batch_docs = []
                if len(ret) % 1000 == 0:
                    logger.info(f"Vectorized {len(ret)} documents...")
        vectorize_doc(batch_docs, target_fields, bert_cls_field, vectorizer)
        ret.extend(batch_docs)
    return ret


def post_documents(es: "es_wrapper.ES", index: str, docs: List[Dict[str, Any]], bulk_size: int):
    es.index(index, docs, bulk_size)


def main():
    config_file = utils.get_dir().parent / "config.json"
    utils.set_root_logger()

    conf = utils.read_json_config(config_file)
    mlit_conf = conf["mlit"]
    bert_cls_field = conf["bert_cls_field"]
    index = mlit_conf["es_index"]

    es = es_wrapper.ES([conf["es_url"]])
    create_index(
        es,
        mlit_conf["es_always_clear_index"],
        index,
        mlit_conf["date_field"],
        mlit_conf["date_format"],
        bert_cls_field,
        conf["bert_vec_size"],
    )

    vectorizer = JaVectorizer()
    docs = get_documents(mlit_conf["csv_path"], mlit_conf["target_fields"], bert_cls_field, vectorizer)
    post_documents(es, index, docs, mlit_conf["es_bulk_size"])


if __name__ == "__main__":
    main()
