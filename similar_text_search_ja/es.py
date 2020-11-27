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

    def index(self, index, docs, bulk_size=2000):
        logger = logging.getLogger(__name__)
        body = ""
        for i, doc in enumerate(docs):
            act_index = {}
            act_index["index"] = {"_index": index, "_id": i}
            body += json.dumps(act_index)
            body += "\n"

            body += json.dumps(doc, ensure_ascii=False)
            body += "\n"
            if (i + 1) % bulk_size == 0:
                logger.info(f"Post {i + 1} documents...")
                self.es.bulk(body)
                body = ""
        if body:
            logger.info(f"Post {i + 1} documents...")
            self.es.bulk(body)
        logger.info(f"Indexed {len(docs)} documents")

    def get(self, index, id):
        return self.es.get(index, id)

    def mlt_by_id(self, index, target_fields, query_doc_id, size=100, min_term_freq=1):
        return self.es.search(
            body={
                "query": {
                    "more_like_this": {
                        "fields": target_fields,
                        "like": {"_index": index, "_id": query_doc_id},
                        "min_term_freq": min_term_freq,
                    }
                },
                "size": size,
            },
            index=index,
        )

    def cosine_by_id(self, index, dense_vector_field, query_doc_id, size=100):
        doc = self.es.get(index, query_doc_id)
        return self.es.search(
            body={
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must_not": [{"ids": {"values": [str(query_doc_id)]}}]
                            }
                        },
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{dense_vector_field}') + 1.0",
                            "params": {
                                "query_vector": doc["_source"][dense_vector_field]
                            },
                        },
                    }
                }
            }
        )

    def cosine_by_vector(self, index, dense_vector_field, vector, size=100):
        return self.es.search(
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{dense_vector_field}') + 1.0",
                            "params": {"query_vector": vector},
                        },
                    }
                }
            }
        )

    def get_hit(self, search_result):
        return [hit for hit in search_result["hits"]["hits"]]
