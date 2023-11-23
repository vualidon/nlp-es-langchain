from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
# import dotenv
from collections import defaultdict
# dotenv = dotenv.load_dotenv()
import os
import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

def bm25_search(es,index, inp_question, size):

    bm25 = es.search(
        index=index, 
        body={"query": 
            {"match": {"content": inp_question }}
        }
        , size=size
    )

    return bm25

def sem_search(es,index, model, inp_question, k):

    question_embedding = model.encode(inp_question)

    sem_search = es.search(
        index=index, 
        knn={
            "field": "embeddings",
            "query_vector": question_embedding,
            "k": k,
            "num_candidates": 100
        }
    )   

    return sem_search

def normalize_bm25_scores(bm25):
    max_score = bm25['hits']['max_score']
    for hit in bm25['hits']['hits']:
        hit['_score'] = hit['_score'] / max_score
    return bm25

def get_min_score(rrf_score):
    return min([min(v) for v in rrf_score.values()])

def hybrid_search(elastic_server,index_name, model, inp_question, alpha=0.5, k=10):
    vector_results = sem_search(elastic_server,index_name, model, inp_question, k//2)
    bm25_results = bm25_search(elastic_server, index_name, inp_question, k//2)
    bm25_results = normalize_bm25_scores(bm25_results)

    vector_ids = [hit['_id'] for hit in vector_results['hits']['hits']]
    bm25_ids = [hit['_id'] for hit in bm25_results['hits']['hits']]
    common_ids = set(vector_ids).intersection(set(bm25_ids))
    rrf_score = {}

    for rank, doc_id in enumerate(vector_ids):
        rank += 1
        rrf_score[doc_id] = [(1 - alpha) / rank]

    for rank, doc_id in enumerate(bm25_ids):
        rank += 1
        if doc_id not in vector_ids:
            rrf_score[doc_id] = [alpha / rank]
        else:
            rrf_score[doc_id].append(alpha / rank)

    min_score = get_min_score(rrf_score)

    for doc_id, scores in rrf_score.items():
        if len(scores) == 1:
            rrf_score[doc_id] = [min_score]
    
    rrf_score = {k: sum(v) for k, v in rrf_score.items()}
    rrf_score = {k: v for k, v in sorted(rrf_score.items(), key=lambda item: item[1], reverse=True)}
    return rrf_score


def multi_query_search(elastic_server, index_name, model, inp_questions, alpha=0.5):

    results = defaultdict(int)
    rrf_scores = []

    for inp_question in inp_questions:
        result = hybrid_search(elastic_server,index_name, model, inp_question, alpha)
        rrf_scores.append(result)

    for rrf_score in rrf_scores:
        for k, v in rrf_score.items():
            results[k] += v

    return dict(sorted(dict(results).items(),key=lambda x: x[1], reverse=True))

def get_docs(es, index_name, rrf_scores, k=5):

    ids = list(rrf_scores.keys())[:k]
    query = {
        "query": {
            "terms": {
                "_id": ids
            }
        }
    }
    return es.search(index=index_name, body=query)['hits']['hits']




# if __name__=="__main__":
#     es = Elasticsearch("http://localhost:9200") 
#     # model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')
#     model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')
#     # index_name = 'xquad_bert_70_cased'
#     index_name = 'xquad_bert_70_cased'
#     inp_question = "Quy chế tuyển sinh của trường Đại học công an nhân dân"

#     inp_questions = generate_questions(inp_question)
#     print(inp_questions)
#     # print(hybrid_search(index_name, model, inp_question, 0.5))

#     rrf_scores = multi_query_search(index_name, model, inp_questions, alpha=0.5)
#     print(rrf_scores)
#     # print(get_docs(index_name, rrf_scores, k=5))
#     docs = get_docs(index_name, rrf_scores, k=5)
#     document = "\n\n".join([doc['_source']['content'] for doc in docs])
#     print(generate_answer(document, inp_question))
    # for doc in docs:

    # for rrfscore in rrf_scores:
    #     print(rrfscore)



# def get_min_score(common_elements, elements_dictionary):
#     if len(common_elements):
#         return min([min(v) for v in elements_dictionary.values()])
#     else:
#         # No common results - assign arbitrary minimum score value
#         return 0.01
    
