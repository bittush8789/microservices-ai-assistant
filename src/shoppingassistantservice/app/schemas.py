from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class PriceUSD(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    currency_code: str = Field(default="USD", alias="currencyCode")
    units: int
    nanos: int
    amount: float
    formatted: str

class Product(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    description: str
    picture: str
    price_usd: PriceUSD = Field(alias="priceUsd")
    categories: List[str]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's query or message")
    image: Optional[str] = Field(default=None, description="Optional image URL or base64 data")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Optional previous conversation messages")

class ChatResponse(BaseModel):
    content: str = Field(..., description="The AI assistant's text response")
    products: List[Product] = Field(default_factory=list, description="Related products matching the user's query")
    extracted_ids: List[str] = Field(default_factory=list, description="Product IDs identified in response")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata or tool execution details")

class HealthResponse(BaseModel):
    status: str
    service: str
    openai_configured: bool
    model: str
    catalog_products_count: int
    rag_mode: Optional[str] = None
    rag_documents_count: Optional[int] = None
    hybrid_rag_enabled: bool = True

# Hybrid RAG Schemas
class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Query text to search for in product knowledge base")
    n_results: int = Field(default=3, ge=1, le=10, description="Number of results to retrieve")
    strategy: Optional[str] = Field(default="hybrid", description="Search strategy: 'hybrid', 'dense' (Chroma), or 'sparse' (BM25)")
    dense_weight: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Weight for dense vs sparse retrieval in hybrid mode (0.0 to 1.0)")

class RAGQueryResult(BaseModel):
    id: str
    name: str
    document: str
    price: str
    distance: float = 0.0
    strategy: str = "hybrid"
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    sparse_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGIndexResponse(BaseModel):
    status: str
    collection: str
    indexed_count: int
    mode: str
    embedding_model: str
    hybrid_enabled: bool = True
    bm25_documents_count: int

class RAGStatusResponse(BaseModel):
    status: str
    mode: str
    host: str
    port: int
    collection: str
    document_count: int
    embedding_model: str
    hybrid_enabled: bool = True
    bm25_documents_count: int
