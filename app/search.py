from elasticsearch import Elasticsearch

# Kết nối tới Elasticsearch
es = Elasticsearch("http://localhost:9200")


mapping = es.indices.get_mapping(index="face-embeddings").body
import json
print(json.dumps(mapping, indent=2, ensure_ascii=False))


# Truy vấn tất cả dữ liệu trong index "face-embeddings"
response = es.search(index="face-embeddings", body={
    "query": {
        "match_all": {}
    },
    "size": 10000  # Số lượng kết quả trả về
})

# In kết quả và số chiều của embedding
for hit in response['hits']['hits']:
    source = hit["_source"]
    embedding = source.get("embedding", [])
    print(f"User ID: {source.get('user_id')}, Name: {source.get('user_name')}, Embedding dim: {len(embedding)}")
