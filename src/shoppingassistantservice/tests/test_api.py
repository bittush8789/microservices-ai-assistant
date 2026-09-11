import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add shoppingassistantservice to sys.path
service_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(service_dir))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "shoppingassistantservice"
    assert data["catalog_products_count"] == 9

def test_root_info_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data
    assert "products_url" in data

def test_list_all_products():
    response = client.get("/products")
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 9
    assert any(p["name"] == "Sunglasses" for p in products)

def test_filter_products_by_category():
    response = client.get("/products?category=accessories")
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 2
    names = [p["name"] for p in products]
    assert "Sunglasses" in names
    assert "Watch" in names

def test_filter_products_by_price():
    response = client.get("/products?max_price=20.0")
    assert response.status_code == 200
    products = response.json()
    for p in products:
        assert p["priceUsd"]["amount"] <= 20.0

def test_get_product_by_id():
    response = client.get("/products/OLJCESPC7Z")
    assert response.status_code == 200
    p = response.json()
    assert p["id"] == "OLJCESPC7Z"
    assert p["name"] == "Sunglasses"
    assert p["priceUsd"]["formatted"] == "$19.99"
    assert p["priceUsd"]["amount"] == 19.99

def test_get_nonexistent_product():
    response = client.get("/products/INVALID_ID")
    assert response.status_code == 404

def test_chat_empty_message_validation():
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400

def test_chat_offline_catalog_mode():
    response = client.post("/chat", json={"message": "How much are the sunglasses?"})
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "Sunglasses" in data["content"] or "OLJCESPC7Z" in data["content"]
    assert "extracted_ids" in data
    assert "OLJCESPC7Z" in data["extracted_ids"]

def test_chat_frontend_compatibility():
    # Frontend calls POST / with {"message": "..."} and expects {"content": "..."}
    response = client.post("/", json={"message": "Show me accessories"})
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert isinstance(data["content"], str)

def test_chat_with_mocked_openai():
    # Mock OpenAI client response
    mock_choice = MagicMock()
    mock_choice.message.content = "Our Sunglasses [OLJCESPC7Z] are available for $19.99!"
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch("app.assistant.assistant._get_client", return_value=mock_openai_client):
        response = client.post("/chat", json={"message": "What is the price of sunglasses?"})
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "OLJCESPC7Z" in data["extracted_ids"]
        assert len(data["products"]) >= 1
        assert data["products"][0]["id"] == "OLJCESPC7Z"
        assert data["products"][0]["priceUsd"]["formatted"] == "$19.99"
