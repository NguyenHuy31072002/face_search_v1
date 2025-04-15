# app/utils/elasticsearch_client.py
from elasticsearch import Elasticsearch

def get_es_client():
    return Elasticsearch(
        hosts=["http://localhost:9200"],  # Sửa lại nếu chạy ở địa chỉ khác
        verify_certs=False
    )
