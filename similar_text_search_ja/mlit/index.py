from typing import Any, Dict, List

from similar_text_search_ja import es as es_wrapper
from similar_text_search_ja import index as index_base
from similar_text_search_ja import utils, vectorizers


def __get_config():
    config_file = utils.get_dir().parent / "config.json"
    return utils.read_json_config(config_file)


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


def __get_documents(conf: Dict[str, Any]):
    mlit_conf = conf["mlit"]

    return index_base.get_documents(
        mlit_conf["csv_path"],
        mlit_conf["target_fields"],
        conf["bert_cls_field"],
        vectorizers.JaVectorizer(),
    )


def __post_documents(
    es: "es_wrapper.ES", docs: List[Dict[str, Any]], conf: Dict[str, Any]
):
    mlit_conf = conf["mlit"]
    index_base.post_documents(
        es, mlit_conf["es_index"], docs, mlit_conf["es_bulk_size"]
    )


def main():
    utils.set_root_logger()

    conf = __get_config()

    es = es_wrapper.ES([conf["es_url"]])

    __create_index(es, conf)

    __post_documents(es, __get_documents(conf), conf)


if __name__ == "__main__":
    main()
