from similar_text_search_ja import es as es_wrapper
from similar_text_search_ja import index as index_base
from similar_text_search_ja import utils, vectorizers


def main():
    config_file = utils.get_dir().parent / "config.json"
    utils.set_root_logger()

    conf = utils.read_json_config(config_file)
    mlit_conf = conf["mlit"]
    bert_cls_field = conf["bert_cls_field"]
    index = mlit_conf["es_index"]

    es = es_wrapper.ES([conf["es_url"]])
    index_base.create_index(
        es,
        mlit_conf["es_always_clear_index"],
        index,
        mlit_conf["date_field"],
        mlit_conf["date_format"],
        bert_cls_field,
        conf["bert_vec_size"],
    )

    vectorizer = vectorizers.JaVectorizer()
    docs = index_base.get_documents(
        mlit_conf["csv_path"], mlit_conf["target_fields"], bert_cls_field, vectorizer
    )
    index_base.post_documents(es, index, docs, mlit_conf["es_bulk_size"])


if __name__ == "__main__":
    main()
