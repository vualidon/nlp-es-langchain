import json
from elasticsearch import helpers, Elasticsearch
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from elasticsearch import helpers
from datetime import datetime


def create_index(es, index_name):
    es_index = {
        "mappings": {
          "properties": {
            "content": {
              "type": "text"
            },
            "metadata": {
              "type": "object"
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

# Main script
if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")
    model = SentenceTransformer('nlplabtdtu/sbert-70M-cased')

    loader = HuggingFaceDatasetLoader(path='nlplabtdtu/university-dataset', page_content_column='body')
    documents = loader.load()

    _documents = documents[:100]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(_documents)

    index_name = 'xquad_bert_70_cased'
    # Check if the index exists, create it if not
    if not es.indices.exists(index=index_name):
        create_index(es, index_name)

    # Bulk index data
    bulk_index_data(es, index_name, docs)
