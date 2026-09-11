from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.catalog import catalog
from app.rag import rag_service
from app.assistant import assistant
from app.guardrails import guardrail_manager
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Product,
    HealthResponse,
    RAGQueryRequest,
    RAGQueryResult,
    RAGIndexResponse,
    RAGStatusResponse,
    GuardrailValidateRequest,
    GuardrailValidateResponse,
    ScorecardResponse,
    EvalMetric,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure product catalog is indexed into Chroma DB
    try:
        rag_service.index_catalog(force=False)
    except Exception as e:
        print(f"Warning: RAG auto-indexing during startup failed: {e}")
    yield
    # Shutdown

app = FastAPI(
    title="Online Boutique AI Shopping Assistant",
    description="FastAPI microservice providing product information, pricing, and RAG-powered AI recommendations using OpenAI and Chroma DB.",
    version="2.1.0",
    lifespan=lifespan,
)

# Enable CORS for browser and microservice access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint indicating service, OpenAI configuration, and Chroma DB status."""
    try:
        rag_status = rag_service.get_status()
        rag_mode = rag_status["mode"]
        rag_count = rag_status["document_count"]
    except Exception:
        rag_mode = "error"
        rag_count = 0

    return HealthResponse(
        status="healthy",
        service="shoppingassistantservice",
        openai_configured=settings.is_openai_configured,
        model=settings.OPENAI_MODEL,
        catalog_products_count=len(catalog.get_all()),
        rag_mode=rag_mode,
        rag_documents_count=rag_count,
    )

@app.get("/", tags=["Info"])
async def root_info():
    """Root info endpoint providing service links and metadata."""
    return {
        "service": "Online Boutique AI Shopping Assistant",
        "version": "2.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "products_url": "/products",
        "chat_url": "/chat",
        "rag_status_url": "/rag/status",
        "openai_configured": settings.is_openai_configured,
    }

@app.post("/", response_model=ChatResponse, tags=["Assistant"])
@app.post("/chat", response_model=ChatResponse, tags=["Assistant"])
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main conversational AI assistant endpoint.
    Retrieves internal product knowledge and pricing from Chroma DB via RAG,
    and generates an augmented response using OpenAI.
    Compatible with frontend's /bot handler and standalone REST clients.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'message' field cannot be empty.",
        )

    response = await assistant.chat(
        message=request.message,
        history=request.history,
        image=request.image,
    )
    return response

@app.get("/products", response_model=List[Product], tags=["Catalog"])
async def list_products(
    query: Optional[str] = Query(default=None, description="Keyword search in name, description, or categories"),
    category: Optional[str] = Query(default=None, description="Filter by category (e.g. accessories, kitchen)"),
    min_price: Optional[float] = Query(default=None, description="Minimum price in USD", ge=0),
    max_price: Optional[float] = Query(default=None, description="Maximum price in USD", ge=0),
) -> List[Product]:
    """Retrieve all products from the catalog or filter by search query, category, and price range."""
    return catalog.search(query=query, category=category, min_price=min_price, max_price=max_price)

@app.get("/products/{product_id}", response_model=Product, tags=["Catalog"])
async def get_product(product_id: str) -> Product:
    """Retrieve detailed product and pricing information by product ID."""
    product = catalog.get_by_id(product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID '{product_id}' not found in catalog.",
        )
    return product

# RAG Endpoints
@app.get("/rag/status", response_model=RAGStatusResponse, tags=["RAG"])
async def get_rag_status():
    """Check Chroma DB vector database status, mode (Docker container HTTP vs persistent), and indexed document count."""
    try:
        status_data = rag_service.get_status()
        return RAGStatusResponse(**status_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chroma DB status check failed: {str(e)}",
        )

@app.post("/rag/index", response_model=RAGIndexResponse, tags=["RAG"])
async def index_catalog_rag(force: bool = Query(default=False, description="Force re-indexing even if collection has documents")):
    """Extract product and pricing information and store/upsert into both Chroma DB and BM25 for Hybrid RAG."""
    try:
        count = rag_service.index_catalog(force=force)
        status_info = rag_service.get_status()
        return RAGIndexResponse(
            status="success",
            collection=settings.CHROMA_COLLECTION,
            indexed_count=count,
            mode=status_info["mode"],
            embedding_model=status_info["embedding_model"],
            hybrid_enabled=True,
            bm25_documents_count=status_info["bm25_documents_count"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index catalog for Hybrid RAG: {str(e)}",
        )

@app.post("/rag/query", response_model=List[RAGQueryResult], tags=["RAG"])
async def query_rag(request: RAGQueryRequest) -> List[RAGQueryResult]:
    """
    Query the Hybrid RAG engine.
    Supports strategies:
    - 'hybrid': Reciprocal Rank Fusion of Chroma DB dense embeddings + BM25 sparse lexical search (default)
    - 'dense': Pure semantic vector similarity via Chroma DB
    - 'sparse': Pure lexical BM25 term frequency matching
    """
    try:
        results = rag_service.retrieve_context(
            query=request.query,
            n_results=request.n_results,
            strategy=request.strategy or "hybrid",
            dense_weight=request.dense_weight if request.dense_weight is not None else 0.5,
        )
        return [
            RAGQueryResult(
                id=r.get("product_id", ""),
                name=r.get("name", ""),
                document=r.get("document", ""),
                price=r.get("price", ""),
                distance=r.get("distance", 0.0),
                strategy=r.get("strategy", request.strategy or "hybrid"),
                dense_rank=r.get("dense_rank"),
                sparse_rank=r.get("sparse_rank"),
                rrf_score=r.get("rrf_score"),
                sparse_score=r.get("sparse_score"),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query Hybrid RAG: {str(e)}",
        )

@app.post("/guardrails/validate", response_model=GuardrailValidateResponse, tags=["Guardrails"])
async def validate_guardrails(request: GuardrailValidateRequest) -> GuardrailValidateResponse:
    """Validates user input against security, prompt injection, and PII guardrails."""
    res = guardrail_manager.validate_input(request.message)
    return GuardrailValidateResponse(
        is_safe=res.is_safe,
        action=res.action,
        reason=res.reason,
        sanitized_message=res.sanitized_message,
        suggested_pills=res.suggested_pills,
        metadata=res.metadata,
    )

@app.get("/evals/scorecard", response_model=ScorecardResponse, tags=["Evaluations"])
@app.get("/evals/run", response_model=ScorecardResponse, tags=["Evaluations"])
async def run_evaluation_benchmark() -> ScorecardResponse:
    """Executes the quantitative evaluation benchmark and returns a comprehensive scorecard."""
    from evals.evaluator import evaluator
    scorecard = await evaluator.run_full_evaluation()
    metrics_dict = {
        k: EvalMetric(metric_name=k, score=v.get("score", 0.0), total_samples=v.get("total_samples", 0), details=v)
        for k, v in scorecard["metrics"].items()
    }
    return ScorecardResponse(
        benchmark_name=scorecard["benchmark_name"],
        overall_score=scorecard["composite_score"],
        total_test_cases=scorecard["total_test_cases"],
        execution_time_seconds=scorecard["execution_time_seconds"],
        metrics=metrics_dict,
        status=scorecard["status"],
    )
