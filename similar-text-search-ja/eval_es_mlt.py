from pathlib import Path
from typing import Dict, List

import es as es_wrapper
import evaluate
import utils


def get_mlt_results(
    es: "es_wrapper.ES",
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
    es: "es_wrapper.ES",
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
            for hit in es.cosine_by_id(index, dense_vector_field, query_doc_id)["hits"]["hits"]
        ]
        results[query_doc_id] = evaluate.compare_docs(query_doc, similar_docs, ans_field)
    return results


def main():
    utils.set_root_logger()

    config_file = utils.get_dir().parent / "config.json"
    reports_dir = Path("reports/mlit")
    reports_dir.mkdir(parents=True, exist_ok=True)

    conf = utils.read_json_config(config_file)
    mlit_conf = conf["mlit"]
    results = get_cosine_results(
        es_wrapper.ES([conf["es_url"]]),
        mlit_conf["es_index"],
        mlit_conf["test_id_min"],
        mlit_conf["test_id_max"],
        conf["bert_cls_field"],
        mlit_conf["ans_field"],
        mlit_conf["invalid_ans"],
    )

    evaluate.print_cases(reports_dir / "cosine-cases.csv", results)
    evaluate.print_summary(
        reports_dir / "cosine-summary.txt", results.values(), [1, 3, 5, 10, 20, 50, 100]
    )


if __name__ == "__main__":
    main()
