from pathlib import Path

import es as es_wrapper
import evaluate
import utils


def main():
    utils.read_log_config()
    app_conf = utils.get_app_config()["DEFAULT"]
    es_url = app_conf["EsUrl"]
    index = app_conf["EsIndex"]
    test_id_min = int(app_conf["TestIdMin"])
    test_id_max = int(app_conf["TestIdMax"])
    target_fields = app_conf["TargetFields"].split(",")
    ans_field = app_conf["AnsField"]

    es = es_wrapper.ES([es_url])

    results = []
    test_ids = range(test_id_max, test_id_min, -1)
    for query_doc_id in test_ids:
        query_doc = es.get(index, query_doc_id)["_source"]
        mlt_docs = [
            hit["_source"]
            for hit in es.mlt_by_id(index, target_fields, query_doc_id)["hits"]["hits"]
        ]
        results.append(evaluate.compare_docs(query_doc, mlt_docs, ans_field))

    reports_dir = Path("reports/mlit")
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluate.print_cases(reports_dir / "mlt-cases.csv", test_ids, results)
    evaluate.print_summary(reports_dir / "mlt-summary.txt", results, [1, 3, 5, 10, 20, 50, 100])


if __name__ == "__main__":
    main()
