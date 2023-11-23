import os
import json

import dotenv
dotenv = dotenv.load_dotenv()
hf_token = os.getenv("HF_TOKEN")

from huggingface_hub import login
login(token=hf_token)

from datasets import load_dataset
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.retrievers import MultiQueryRetriever, MultiVectorRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from elasticsearch import helpers
from datetime import datetime

from query_transformation import llm_chain

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


embedding = HuggingFaceEmbeddings(model_name='nlplabtdtu/sbert-70M-cased')
elastic_vector_search = ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name="university",
            embedding=embedding
        )

retriever = MultiQueryRetriever(
    retriever=elastic_vector_search.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)

vector_retriever = MultiVectorRetriever(
    vectorstore=elastic_vector_search,
    docstore=InMemoryStore(),
    llm_chain=llm_chain, parser_key="lines"
)

def bulk_data_hf(index_name, dataset_name="nlplabtdtu/university-dataset"):
    embedding = HuggingFaceEmbeddings(model_name='nlplabtdtu/sbert-70M-cased')
    loader = HuggingFaceDatasetLoader(path=dataset_name, page_content_column='body')
    documents = loader.load()

    _documents = documents[:100]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(_documents)
    
    db = ElasticsearchStore.from_documents(
        docs,
        embedding,
        es_url="http://localhost:9200",
        index_name=index_name,
    )

    db.client.indices.refresh(index=index_name)


if __name__=="__main__":
    query = "Trường đại học Công an nhân dân xét tuyển các hình thức nào?"
    # bulk_data_hf(index_name="university")
    unique_docs = retriever.get_relevant_documents(
        query=query
    )

    ans = retriever.vectorstore.similarity_search(query)[0]
    print(ans)
    # print(unique_docs[0])
    
