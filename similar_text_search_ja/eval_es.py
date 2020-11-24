import sys
from pathlib import Path
from typing import Dict, List

from similar_text_search_ja import config, evaluate, utils
from similar_text_search_ja.config import Config
from similar_text_search_ja.es import ES


def get_mlt_results(
    es: "ES",
    index: str,
    test_id_min: int,
    test_id_max: int,
    target_fields: List[str],
    ans_field: str,
    invalid_ans: str,
) -> Dict[int, List[bool]]:
    results = {}
    for query_doc_id in range(test_id_max, test_id_min, -1):
        query_doc = es.get(index, query_doc_id)["_source"]
        ans = query_doc[ans_field]
        if not ans or ans == invalid_ans:
            continue

        mlt_docs = [
            hit["_source"]
            for hit in es.mlt_by_id(index, target_fields, query_doc_id)["hits"]["hits"]
        ]
        results[query_doc_id] = evaluate.compare_docs(query_doc, mlt_docs, ans_field)
    return results


def get_cosine_results(
    es: "ES",
    index: str,
    test_id_min: int,
    test_id_max: int,
    dense_vector_field: str,
    ans_field: str,
    invalid_ans: str,
) -> Dict[int, List[bool]]:
    results = {}
    for query_doc_id in range(test_id_max, test_id_min, -1):
        query_doc = es.get(index, query_doc_id)["_source"]
        ans = query_doc[ans_field]
        if not ans or ans == invalid_ans:
            continue

        similar_docs = [
            hit["_source"]
            for hit in es.cosine_by_id(index, dense_vector_field, query_doc_id)["hits"][
                "hits"
            ]
        ]
        results[query_doc_id] = evaluate.compare_docs(
            query_doc, similar_docs, ans_field
        )
    return results


def print_summary(reports_dir: Path, results: Dict[int, List[bool]]):
    evaluate.print_cases(reports_dir / "sentence-bert-cases.csv", results)
    evaluate.print_summary(
        reports_dir / "sentnece-bert-summary.txt",
        results.values(),
        [1, 3, 5, 10, 20, 50, 100],
    )


def main():
    utils.set_root_logger()

    conf = Config(sys.argv[1])
    reports_dir = conf.report_dir
    config.create_dir(reports_dir)

    results = get_cosine_results(
        ES([conf.es_url]),
        conf.es_index_name,
        conf.test_id_min,
        conf.test_id_max,
        conf.es_embedding_field,
        conf.ans_field,
        conf.invalid_ans,
    )

    print_summary(reports_dir, results)


if __name__ == "__main__":
    main()
