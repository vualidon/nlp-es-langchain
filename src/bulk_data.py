import json
from elasticsearch import helpers, Elasticsearch
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from elasticsearch import helpers
from datetime import datetime
from read_pdf import *

def create_index(es, index_name):
    es_index = {
        "mappings": {
          "properties": {
            "content": {
              "type": "text"
            },
            "metadata": {
              "type": "text"
            },
            "date": {
              "type": "date"
            },
            "embeddings": {
              "type": "dense_vector",
              "dims": 768
            }
          }
        }
      }

    es.indices.create(index=index_name, body=es_index)

def bulk_index_data(es, index_name, docs):
    data = []

    for doc in docs:
        embeddings = model.encode(doc.page_content)
        data.append({
            '_index': index_name,
            '_source': {
                'content': doc.page_content,
                'embeddings': embeddings,
                'metadata': doc.metadata,
                'date': datetime.now()
            }
        })

    helpers.bulk(es, data)

def bulk_index_law_data(es, index_name, docs):
    data = []

    for doc in docs:
        embeddings = model.encode(doc)
        data.append({
            '_index': index_name,
            '_source': {
                'content': doc,
                'embeddings': embeddings,
                'metadata': "luat_ma_tuy",
                'date': datetime.now()
            }
        })

    helpers.bulk(es, data)

def add_data(es, index_name, doc):
    embeddings = model.encode(doc)
    data = {
        'content': doc,
        'embeddings': embeddings,
        'metadata': "luat_ma_tuy",
        'date': datetime.now()
    }
    res = es.index(index=index_name, body=data)
    # print(res)
  
def add_datas(es, index_name, docs):
    for doc in docs:
        add_data(es, index_name, doc)

# Main script
if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200",) # timeout=600, max_retries=10, retry_on_timeout=True)
    model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')

    # loader = HuggingFaceDatasetLoader(path='nlplabtdtu/university-dataset', page_content_column='body')
    # documents = loader.load()

    # _documents = documents[:100]

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    # docs = text_splitter.split_documents(_documents)
    docs = split_document('./law_data/1.pdf')
    index_name = 'luat_ma_tuy'
    # Check if the index exists, create it if not
    if not es.indices.exists(index=index_name):
        create_index(es, index_name)

    # Bulk index data
    add_datas(es, index_name, docs)
