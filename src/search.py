from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')

def bm25_search(query, n):
    bm25 = es.search(
        index="edu_data", 
        body={"query": 
            {"match": {"content": query }}
        }
    )

    return bm25['hits']['hits'][0:n]

def semantic_search(query, n):
    question_embedding = model.encode(query)

    sem_search = es.search(index="edu_data", body=
                        {
                                "query": {
                                    "script_score": {
                                        "query" : {
                                            "match_all": {},
                                        },
                                        "script": {
                                            "source": "cosineSimilarity(params.query_vector, 'content_embedding') + 1.0", 
                                            "params": {
                                                "query_vector": question_embedding
                                            }
                                        }
                                    }
                                }
                            }
    )

    return sem_search['hits']['hits'][0:n]