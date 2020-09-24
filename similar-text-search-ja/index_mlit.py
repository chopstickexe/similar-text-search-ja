import argparse

import es as es_wrapper
import utils
from mlit import parser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        help="Delete & re-create the index if exists",
    )
    default_bulk_size = 5000
    parser.add_argument(
        "-s",
        "--bulk_size",
        type=int,
        default=default_bulk_size,
        help=f"Bulk index size (Default={default_bulk_size})",
    )
    return parser.parse_args()


def main():
    utils.read_log_config()
    app_conf = utils.get_app_config()["DEFAULT"]
    es_url = app_conf["EsUrl"]
    index = app_conf["EsIndex"]
    csv_path = app_conf["CsvPath"]

    args = get_args()

    es = es_wrapper.ES([es_url])
    if args.clear:
        es.delete_index(index)
    es.create_index(
        index,
        body={
            "mappings": {
                "properties": {"受付日": {"type": "date", "format": "yyyy年MM月dd日"}}
            }
        },
    )
    with parser.Parser(csv_path) as mlit_parser:
        es.index(index, mlit_parser, args.bulk_size)


if __name__ == "__main__":
    main()
