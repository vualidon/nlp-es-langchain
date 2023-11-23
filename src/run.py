from search import *
from generate import *
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

def main():
    es = Elasticsearch("http://localhost:9200") 
    # model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')
    model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')
    # index_name = 'xquad_bert_70_cased'
    index_name = 'xquad_bert_70_cased'
    inp_question = "Quy chế tuyển sinh của trường Đại học công an nhân dân"

    inp_questions = generate_questions(inp_question)
    print(inp_questions)

    rrf_scores = multi_query_search(es, index_name, model, inp_questions, alpha=0.5)
    print(rrf_scores)
    docs = get_docs(es, index_name, rrf_scores, k=5)
    document = "\n\n".join([doc['_source']['content'] for doc in docs])
    print(generate_answer(document, inp_question))

main()
