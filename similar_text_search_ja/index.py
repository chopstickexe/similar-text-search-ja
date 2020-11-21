import csv
import logging
from typing import Any, Dict, List

from similar_text_search_ja import csv_parser
from similar_text_search_ja import es as es_wrapper
from similar_text_search_ja import utils, vectorizers


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


def get_documents(
    csv_path: str,
    target_fields: List[str],
    vector_field: str,
    vectorizer: "vectorizers.JaVectorizer",
) -> List[Dict[str, Any]]:
    """Read documents from a csv file and add a dense vector to each doc

    Args:
        csv_path (str): CSV path
        target_fields (List[str]): Text fields
        vector_field (str): Field to store dense vectors representing the text
        vectorizer (vectorizers.JaVectorizer): Vectorizer instance

    Returns:
        List[Dict[str, Any]]: Documents with dense vectors
    """
    logger = logging.getLogger(__name__)
    ret = []
    with csv_parser.CsvParser(csv_path) as parser:
        batch_docs = []
        for doc in parser:
            batch_docs.append(doc)
            if len(batch_docs) == 32:
                _vectorize_doc(batch_docs, target_fields, vector_field, vectorizer)
                ret.extend(batch_docs)
                batch_docs = []
                if len(ret) % 1000 == 0:
                    logger.info(f"Vectorized {len(ret)} documents...")
        _vectorize_doc(batch_docs, target_fields, vector_field, vectorizer)
        ret.extend(batch_docs)
    return ret


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


def add_vectors(
    docs: List[Dict[str, Any]],
    target_fields: List[str],
    vector_field: str,
    vectorizer: "vectorizers.JaVectorizer",
):
    _vectorize_doc(docs, target_fields, vector_field, vectorizer)


def _vectorize_doc(batch_docs, target_fields, bert_cls_field, vectorizer):
    if not batch_docs or len(batch_docs) == 0:
        return
    batch_txts = [_get_target_fields_txt(doc, target_fields) for doc in batch_docs]
    input_ids = vectorizer.encode(batch_txts)
    outputs = vectorizer.vectorize(input_ids)
    assert len(outputs.last_hidden_state) == len(batch_docs)
    for i, doc in enumerate(batch_docs):
        doc[bert_cls_field] = outputs.last_hidden_state[i][0][:].tolist()


def _get_target_fields_txt(doc: Dict[str, Any], target_fields: List[str]) -> str:
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


def main():
    """Example usage"""
    utils.set_root_logger()
    logger = logging.getLogger(__name__)

    es = es_wrapper.ES(["es01:9200"])
    fields = {
        "受付日": {
            "type": "date",
            "format": "yyyy年MM月dd日",
        },
        "bert_cls_vec": {
            "type": "dense_vector",
            "dims": 768,
        },
    }
    index = "test"
    create_index(es=es, clear_index=True, index=index, fields=fields)

    docs = []
    for batch in doc_generator("data/mlit/mlit.sample.csv", batch_size=100):
        logger.info(f"Vectorize {len(batch)} documents...")
        add_vectors(
            batch,
            target_fields=["申告内容の要約"],
            vector_field="bert_cls_vec",
            vectorizer=vectorizers.JaVectorizer(),
        )
        docs.extend(batch)

    post_documents(es=es, index=index, docs=docs, bulk_size=1000)
    es.delete_index(index)


if __name__ == "__main__":
    main()
