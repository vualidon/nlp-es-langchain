from elasticsearch import Elasticsearch

# Replace with your Elasticsearch host and port
es = Elasticsearch(hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

try:
    # Check if Elasticsearch is up and running
    if es.ping():
        print('Connected to Elasticsearch')
    else:
        print('Failed to connect to Elasticsearch')
except Exception as e:
    print(f'Error connecting to Elasticsearch: {e}')
