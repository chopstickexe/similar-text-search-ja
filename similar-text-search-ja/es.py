import json
import logging

from elasticsearch import Elasticsearch


class ES:
    def __init__(self, urls):
        self.es = Elasticsearch(urls)

    def delete_index(self, index):
        if self.es.indices.exists(index):
            self.es.indices.delete(index)

    def create_index(self, index, body):
        self.es.indices.create(index, body=body)

    def index(self, index, parser, bulk_size=2000):
        logger = logging.getLogger(__name__)
        body = ""

        for i, row in enumerate(parser):
            act_index = {}
            act_index["index"] = {"_index": index, "_id": i}
            body += json.dumps(act_index)
            body += "\n"

            doc = row
            body += json.dumps(doc, ensure_ascii=False)
            body += "\n"
            if (i + 1) % bulk_size == 0:
                logger.info(f"Post {i + 1} documents...")
                self.es.bulk(body)
                body = ""
        if body:
            self.es.bulk(body)
        logger.info(f"Indexed {i + 1} documents")
