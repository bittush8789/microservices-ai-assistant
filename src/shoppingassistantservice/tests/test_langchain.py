import json
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.main import app
from app.config import settings
from app.rag import rag_service, LangChainHybridRetriever
from app.guardrails import guardrail_manager
from app.assistant import (
    assistant,
    LANGCHAIN_TOOLS,
    TOOL_MAP,
    get_product_details,
    get_product_pricing,
    search_products,
    search_knowledge_base,
)

client = TestClient(app)

def test_langchain_tools_definition_and_execution():
    tool_names = [t.name for t in LANGCHAIN_TOOLS]
    assert "get_product_details" in tool_names
    assert "get_product_pricing" in tool_names
    assert "search_products" in tool_names
    assert "search_knowledge_base" in tool_names

    # Test get_product_details
    details_json = get_product_details.invoke({"product_id_or_name": "OLJCESPC7Z"})
    details = json.loads(details_json)
    assert details["id"] == "OLJCESPC7Z"
    assert "Sunglasses" in details["name"]

    # Test get_product_pricing
    pricing_json = get_product_pricing.invoke({"product_id_or_name": "OLJCESPC7Z"})
    pricing = json.loads(pricing_json)
    assert pricing["id"] == "OLJCESPC7Z"
    assert pricing["price"] == "$19.99"
    assert pricing["currency"] == "USD"

    # Test search_products
    search_json = search_products.invoke({"category": "accessories"})
    results = json.loads(search_json)
    assert len(results) > 0
    assert any(p["id"] == "OLJCESPC7Z" for p in results)

    # Test search_knowledge_base
    rag_json = search_knowledge_base.invoke({"query": "hairdryer voltage", "strategy": "hybrid"})
    rag_results = json.loads(rag_json)
    assert len(rag_results) > 0
    assert rag_results[0]["product_id"] == "2ZYFJ3GM2N"

def test_langchain_hybrid_retriever_document_conversion():
    retriever = rag_service.as_langchain_retriever(n_results=2, strategy="hybrid")
    assert isinstance(retriever, LangChainHybridRetriever)

    docs = retriever.invoke("hairdryer")
    assert len(docs) > 0
    assert len(docs) <= 2

    top_doc = docs[0]
    assert isinstance(top_doc, Document)
    assert top_doc.metadata["product_id"] == "2ZYFJ3GM2N"
    assert top_doc.metadata["strategy"] == "hybrid"
    assert "Hairdryer" in top_doc.metadata["name"]
    assert len(top_doc.page_content) > 0

def test_langchain_model_tool_binding():
    # Instantiate test ChatOpenAI instance
    test_llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-api-key", temperature=0.7)
    bound_llm = test_llm.bind_tools(LANGCHAIN_TOOLS)
    assert bound_llm is not None

def test_langsmith_status_endpoint():
    response = client.get("/langsmith/status")
    assert response.status_code == 200
    data = response.json()
    assert "tracing_enabled" in data
    assert "endpoint" in data
    assert "project" in data
    assert "langchain_version" in data
    assert "langsmith_version" in data
    assert data["project"] == "online-boutique-shopping-assistant"

def test_health_endpoint_includes_langchain_and_langsmith():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "langchain_version" in data
    assert "langsmith_enabled" in data
    assert "langsmith_project" in data
    assert data["langchain_version"] == "0.3.26"

def test_langsmith_traceable_guardrails_and_rag():
    # Test that @traceable wrappers execute normally without errors
    guard_res = guardrail_manager.validate_input("Show me sunglasses")
    assert guard_res.is_safe is True

    context = rag_service.retrieve_context("sunglasses", n_results=2)
    assert len(context) > 0
