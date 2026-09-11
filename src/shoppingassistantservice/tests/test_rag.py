import pytest
import sys
from pathlib import Path

# Add shoppingassistantservice to sys.path
service_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(service_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.rag import rag_service

client = TestClient(app)

def test_rag_service_initialization():
    status = rag_service.get_status()
    assert status["status"] == "connected"
    assert status["collection"] == "online_boutique_products"
    assert status["document_count"] >= 9
    assert status["hybrid_enabled"] is True
    assert status["bm25_documents_count"] >= 9

def test_rag_index_endpoint():
    response = client.post("/rag/index?force=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["indexed_count"] == 9
    assert data["collection"] == "online_boutique_products"
    assert data["hybrid_enabled"] is True
    assert data["bm25_documents_count"] == 9

def test_rag_status_endpoint():
    response = client.get("/rag/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "connected"
    assert data["collection"] == "online_boutique_products"
    assert data["document_count"] == 9
    assert data["hybrid_enabled"] is True
    assert data["bm25_documents_count"] == 9
    assert "mode" in data

def test_bm25_sparse_retrieval():
    # Exact code and spec lookup via BM25
    sparse_res = rag_service.retrieve_context("UV400 aviator polarized", strategy="sparse", n_results=1)
    assert len(sparse_res) > 0
    assert sparse_res[0]["product_id"] == "OLJCESPC7Z" # Sunglasses
    assert sparse_res[0]["sparse_score"] > 0

    watt_res = rag_service.retrieve_context("1800W travel 240V", strategy="sparse", n_results=1)
    assert len(watt_res) > 0
    assert watt_res[0]["product_id"] == "2ZYFJ3GM2N" # Hairdryer

def test_dense_semantic_retrieval():
    dense_res = rag_service.retrieve_context("eye protection for sunny days", strategy="dense", n_results=2)
    assert len(dense_res) > 0
    ids = [r["product_id"] for r in dense_res]
    assert "OLJCESPC7Z" in ids

def test_hybrid_rag_reciprocal_rank_fusion():
    # Hybrid search uses both Chroma DB and BM25 with RRF scoring
    hybrid_res = rag_service.retrieve_context("compact travel hairdryer dual voltage", strategy="hybrid", n_results=3)
    assert len(hybrid_res) > 0
    top = hybrid_res[0]
    assert top["product_id"] == "2ZYFJ3GM2N" # Hairdryer
    assert top["strategy"] == "hybrid"
    assert top["rrf_score"] is not None
    assert top["rrf_score"] > 0.0
    assert top["dense_rank"] is not None
    assert top["sparse_rank"] is not None

def test_rag_query_endpoint_strategies():
    # Test hybrid strategy (default)
    res_hybrid = client.post("/rag/query", json={"query": "borosilicate glass bamboo lid storage", "strategy": "hybrid", "n_results": 2})
    assert res_hybrid.status_code == 200
    data_h = res_hybrid.json()
    assert len(data_h) > 0
    assert data_h[0]["id"] == "9SIQT8TOJO"
    assert data_h[0]["strategy"] == "hybrid"
    assert data_h[0]["rrf_score"] is not None

    # Test sparse strategy
    res_sparse = client.post("/rag/query", json={"query": "UV400", "strategy": "sparse", "n_results": 1})
    assert res_sparse.status_code == 200
    data_s = res_sparse.json()
    assert len(data_s) > 0
    assert data_s[0]["id"] == "OLJCESPC7Z"
    assert data_s[0]["strategy"] == "sparse"

    # Test dense strategy
    res_dense = client.post("/rag/query", json={"query": "timepiece accessory", "strategy": "dense", "n_results": 2})
    assert res_dense.status_code == 200
    data_d = res_dense.json()
    assert len(data_d) > 0
    assert data_d[0]["strategy"] == "dense"

def test_chat_uses_rag_context():
    response = client.post("/chat", json={"message": "What is the warranty and voltage on the hairdryer?"})
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "2ZYFJ3GM2N" in data["extracted_ids"]
    assert "details" in data
    assert "rag_mode" in data["details"]
    assert data["details"]["rag_results_count"] > 0
