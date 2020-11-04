import es as es_wrapper
import utils
from mlit import parser


def create_index(
    es: "es_wrapper.ES",
    clear_index: bool,
    index: str,
    date_field: str,
    date_format: str,
):
    if clear_index:
        es.delete_index(index)

    es.create_index(
        index,
        body={
            "mappings": {
                "properties": {date_field: {"type": "date", "format": date_format}}
            }
        },
    )


def post_documents(es: "es_wrapper.ES", index: str, csv_path: str, bulk_size: int):
    with parser.Parser(csv_path) as mlit_parser:
        es.index(index, mlit_parser, bulk_size)


def main():
    config_file = utils.get_dir().parent / "config.json"
    utils.set_root_logger()

    conf = utils.read_json_config(config_file)
    mlit_conf = conf["mlit"]
    index = mlit_conf["es_index"]

    es = es_wrapper.ES([conf["es_url"]])
    create_index(
        es,
        mlit_conf["es_always_clear_index"],
        index,
        mlit_conf["date_field"],
        mlit_conf["date_format"],
    )
    post_documents(es, index, mlit_conf["csv_path"], mlit_conf["es_bulk_size"])


if __name__ == "__main__":
    main()
