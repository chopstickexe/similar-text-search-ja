import argparse
import json
import logging

from elasticsearch import Elasticsearch

from mlit import parser


def get_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Elasticsearch URL (hostname:port)")
    parser.add_argument("index", help="Elasticsearch index name")
    parser.add_argument("csv_path", help="path to the mlit csv")
    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        help="Delete & re-create the index if exists",
    )
    default_bulk_size = 2000
    parser.add_argument(
        "-s",
        "--bulk_size",
        type=int,
        default=default_bulk_size,
        help=f"Bulk index size (Default={default_bulk_size})",
    )
    return parser.parse_args()


def main():
    logger = get_logger(__name__)
    args = get_args()

    es = Elasticsearch([args.url])
    if args.clear and es.indices.exists(args.index):
        es.indices.delete(args.index)
    es.indices.create(
        args.index,
        body={
            "mappings": {
                "properties": {"受付日": {"type": "date", "format": "yyyy年MM月dd日"}}
            }
        },
    )

    body = ""
    with parser.Parser("data/mlit/mlit.20200919.csv") as mlit_parser:
        for i, row in enumerate(mlit_parser):
            act_index = {}
            act_index["index"] = {"_index": args.index, "_id": i}
            body += json.dumps(act_index)
            body += "\n"

            doc = row
            body += json.dumps(doc, ensure_ascii=False)
            body += "\n"
            if (i + 1) % args.bulk_size == 0:
                logger.info(f"Post {i + 1} documents...")
                es.bulk(body)
                body = ""
        if body:
            es.bulk(body)
        logger.info(f"Indexed {i + 1} documents")


if __name__ == "__main__":
    main()
