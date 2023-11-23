from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
es.indices.delete(index="luat_ma_tuy")
print("Done")