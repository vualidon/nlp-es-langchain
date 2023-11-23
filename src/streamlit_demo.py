import streamlit as st
from search import *
from generate import *
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import tiktoken

st.title("# DEMO")
inp_question = st.text_input("Enter your question here:")
btn_submit = st.button("SUBMIT")

es = Elasticsearch("http://localhost:9200") 
model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')
index_name = 'xquad_bert_70_cased'



if btn_submit:
    inp_questions = generate_questions(inp_question)

    rrf_scores = multi_query_search(es, index_name, model, inp_questions, alpha=0.5)
    docs = get_docs(es, index_name, rrf_scores, k=5)
    document = "\n\n".join([doc['_source']['content'] for doc in docs])
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    print("TOKENS", len(encoding.encode(document)))
    with open("document.txt", "w", encoding='utf8') as f:
        f.write(document)
    answer = generate_answer(document, inp_question)
    st.write(answer)