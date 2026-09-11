import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from app.config import settings
from app.catalog import catalog
from app.schemas import Product

logger = logging.getLogger("rag_service")

# Internal enriched specifications for Online Boutique products
INTERNAL_PRODUCT_SPECS = {
    "OLJCESPC7Z": {
        "material": "Lightweight alloy metal frame with acetate temple tips",
        "features": "UV400 protection, polarized glare-reduction lenses, classic aviator teardrop design",
        "dimensions": "Lens width: 58mm, Bridge: 14mm, Arm: 140mm",
        "use_case": "Everyday outdoor wear, driving, beach, modern styling",
        "warranty": "1-year limited manufacturer warranty",
    },
    "66VCHSJNUP": {
        "material": "100% combed organic cotton (180 GSM)",
        "features": "Scoop neckline, double-stitched hem, breathable soft-wash fabric, cropped silhouette",
        "care": "Machine wash cold inside out, tumble dry low, do not bleach",
        "use_case": "Warm weather casual styling, layering, loungewear",
        "warranty": "30-day satisfaction guarantee",
    },
    "1YMWWN1N4O": {
        "material": "Gold-tone ion-plated stainless steel case and link bracelet",
        "features": "Japanese quartz three-hand movement, mineral crystal scratch-resistant glass, water-resistant to 30 meters (3 ATM)",
        "dimensions": "Case diameter: 40mm, Band width: 20mm",
        "use_case": "Business casual, formal events, daily timekeeping",
        "warranty": "2-year international warranty",
    },
    "L9ECAV7KIM": {
        "material": "Supple genuine leather upper with breathable leather lining and rubber sole",
        "features": "Slip-on penny loafer silhouette, ergonomic cushioned memory foam footbed, anti-slip tread",
        "care": "Wipe with damp cloth; apply leather conditioner periodically",
        "use_case": "Summer smart casual, office, warm weather evening gatherings",
        "warranty": "6-month craftsmanship guarantee",
    },
    "2ZYFJ3GM2N": {
        "material": "High-impact heat-resistant polycarbonate casing",
        "features": "1800W DC motor, tourmaline-infused negative ion generator (anti-frizz), 3 heat and 2 speed settings with cool-shot button, foldable handle for travel, dual voltage 110-240V",
        "dimensions": "Weight: 410g, Cord length: 1.8m",
        "use_case": "Quick drying, salon styling at home, international travel",
        "warranty": "2-year replacement warranty",
    },
    "0PUK6V6EV0": {
        "material": "Wrought iron and heat-tempered ceramic glaze finish",
        "features": "Intricate geometric filigree pattern, stable non-scratch felt base, accommodates tea lights and standard votives up to 2 inches",
        "dimensions": "Height: 12cm, Base diameter: 8cm, Weight: 280g",
        "use_case": "Living room centerpiece, dining ambiance, housewarming gift",
        "warranty": "Lifetime quality assurance against manufacturing defects",
    },
    "LS4PSXUNUM": {
        "material": "Food-grade 304 brushed stainless steel tops with lead-free thick glass bodies",
        "features": "Adjustable multi-size dispensing holes (fine, medium, coarse, closed), moisture-tight seal, transparent base shows fill level",
        "dimensions": "Capacity: 120ml each, Height: 11.5cm",
        "care": "Glass body is dishwasher safe; hand wash lids and dry thoroughly",
        "use_case": "Kitchen seasoning, tabletop dining, outdoor BBQ",
        "warranty": "1-year limited warranty",
    },
    "9SIQT8TOJO": {
        "material": "High-borosilicate thermal-shock-resistant glass with natural sustainable bamboo lid and food-grade silicone airtight ring",
        "features": "Airtight moisture-proof seal keeps food fresh, wide mouth for easy scooping and cleaning, BPA-free",
        "dimensions": "Capacity: 57 oz (1.7 Liters), Height: 24cm, Diameter: 10cm",
        "care": "Glass jar is dishwasher and microwave safe; hand wash bamboo lid",
        "use_case": "Storing pasta, coffee beans, flour, cereal, dry goods",
        "warranty": "1-year replacement warranty",
    },
    "6E92ZMYYFZ": {
        "material": "Heavyweight high-fired ceramic stoneware",
        "features": "Contrasting glossy mustard interior with matte charcoal exterior, ergonomic comfort-grip handle, chip-resistant rim",
        "dimensions": "Capacity: 12 oz (350ml), Height: 9.5cm",
        "care": "Microwave and dishwasher safe",
        "use_case": "Morning coffee, tea, hot chocolate, desk companion",
        "warranty": "1-year chip-free guarantee",
    }
}

class BM25Index:
    """Sparse Lexical Retriever using BM25Okapi."""
    def __init__(self):
        self.doc_ids: List[str] = []
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, dict] = {}
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text) if len(token) > 1]

    def build(self, ids: List[str], documents: List[str], metadatas: List[dict]) -> None:
        self.doc_ids = ids
        self.documents = {i: doc for i, doc in zip(ids, documents)}
        self.metadatas = {i: meta for i, meta in zip(ids, metadatas)}

        tokenized_corpus = [self.tokenize(doc) for doc in documents]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def count(self) -> int:
        return len(self.doc_ids)

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.doc_ids:
            return []

        tokens = self.tokenize(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for rank, idx in enumerate(ranked_indices[:n_results], start=1):
            score = float(scores[idx])
            doc_id = self.doc_ids[idx]
            meta = self.metadatas.get(doc_id, {})
            results.append({
                "product_id": meta.get("product_id", ""),
                "name": meta.get("name", ""),
                "price": meta.get("price", ""),
                "price_amount": meta.get("price_amount", 0.0),
                "document": self.documents.get(doc_id, ""),
                "sparse_score": score,
                "sparse_rank": rank,
                "distance": 0.0,
                "strategy": "sparse",
                "metadata": meta,
            })
        return results

class RAGService:
    def __init__(self):
        self._client: Optional[ClientAPI] = None
        self._collection = None
        self._mode: str = "uninitialized"
        self._bm25 = BM25Index()

    def _init_chroma_client(self) -> ClientAPI:
        # First attempt: Connect to Chroma DB running in Docker over HTTP
        try:
            http_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
            http_client.heartbeat()
            self._mode = "docker_http"
            logger.info(f"Connected to Chroma DB in Docker container at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
            return http_client
        except Exception as e:
            logger.warning(
                f"Could not connect to Chroma DB Docker container at {settings.CHROMA_HOST}:{settings.CHROMA_PORT} ({e}). "
                "Falling back to local persistent Chroma client."
            )

        # Fallback: Connect to local persistent Chroma client
        persist_dir = Path(__file__).resolve().parent / "data" / "chroma_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        local_client = chromadb.PersistentClient(path=str(persist_dir))
        self._mode = "local_persistent"
        return local_client

    def get_client(self) -> ClientAPI:
        if self._client is None:
            self._client = self._init_chroma_client()
        return self._client

    def _get_embedding_function(self):
        if settings.is_openai_configured:
            try:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.OPENAI_API_KEY,
                    model_name=settings.OPENAI_EMBEDDING_MODEL,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAIEmbeddingFunction ({e}), using default embedding function.")
        return embedding_functions.DefaultEmbeddingFunction()

    def get_collection(self):
        if self._collection is None:
            client = self.get_client()
            emb_fn = self._get_embedding_function()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                embedding_function=emb_fn,
                metadata={"description": "Online Boutique internal product knowledge & pricing database"}
            )
        return self._collection

    def _create_product_document(self, product: Product) -> str:
        specs = INTERNAL_PRODUCT_SPECS.get(product.id, {})
        spec_text = "\n".join(f"- {k.capitalize()}: {v}" for k, v in specs.items())

        return f"""PRODUCT RECORD:
Product ID: [{product.id}]
Product Name: {product.name}
Categories: {", ".join(product.categories)}
Price: {product.price_usd.formatted} ({product.price_usd.currency_code})
Price in USD Amount: ${product.price_usd.amount:.2f}

Official Catalog Description:
{product.description}

Internal Technical Specifications & Features:
{spec_text}
"""

    def index_catalog(self, force: bool = False) -> int:
        collection = self.get_collection()
        current_count = collection.count()

        products = catalog.get_all()
        ids = []
        documents = []
        metadatas = []

        for p in products:
            doc_id = f"prod_{p.id}"
            doc_text = self._create_product_document(p)
            meta = {
                "product_id": p.id,
                "name": p.name,
                "price": p.price_usd.formatted,
                "price_amount": float(p.price_usd.amount),
                "primary_category": p.categories[0] if p.categories else "general",
                "categories": ", ".join(p.categories),
            }
            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(meta)

        # Build / re-build BM25 sparse index
        self._bm25.build(ids=ids, documents=documents, metadatas=metadatas)

        # Index into Chroma DB collection
        if current_count < len(products) or force:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Successfully indexed {len(ids)} product knowledge documents in Chroma DB ({self._mode}).")

        logger.info(f"Hybrid RAG: {len(ids)} documents indexed in Chroma DB and BM25.")
        return len(ids)

    def retrieve_dense(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Dense semantic retrieval via Chroma DB vector embeddings."""
        collection = self.get_collection()
        if collection.count() == 0:
            self.index_catalog()

        n = min(n_results, max(1, collection.count()))
        results = collection.query(
            query_texts=[query],
            n_results=n,
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                retrieved.append({
                    "product_id": meta.get("product_id", ""),
                    "name": meta.get("name", ""),
                    "price": meta.get("price", ""),
                    "price_amount": meta.get("price_amount", 0.0),
                    "document": doc,
                    "distance": float(dist) if dist is not None else 0.0,
                    "dense_rank": rank,
                    "strategy": "dense",
                    "metadata": meta,
                })

        return retrieved

    def retrieve_sparse(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Sparse lexical retrieval via BM25."""
        if self._bm25.count() == 0:
            self.index_catalog()
        return self._bm25.query(query_text=query, n_results=n_results)

    def retrieve_hybrid(
        self,
        query: str,
        n_results: int = 3,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Retrieval combining Chroma DB (Dense) and BM25 (Sparse)
        using Reciprocal Rank Fusion (RRF).
        """
        total_items = max(9, n_results * 3)
        dense_candidates = self.retrieve_dense(query, n_results=total_items)
        sparse_candidates = self.retrieve_sparse(query, n_results=total_items)

        dense_map = {c["product_id"]: (c, c["dense_rank"]) for c in dense_candidates}
        sparse_map = {c["product_id"]: (c, c["sparse_rank"]) for c in sparse_candidates}

        all_pids = set(dense_map.keys()).union(sparse_map.keys())
        scored_candidates = []

        sparse_weight = max(0.0, min(1.0, 1.0 - dense_weight))
        d_weight = max(0.0, min(1.0, dense_weight))

        for pid in all_pids:
            dense_info = dense_map.get(pid)
            sparse_info = sparse_map.get(pid)

            dense_rank = dense_info[1] if dense_info else 999
            sparse_rank = sparse_info[1] if sparse_info else 999

            # Reciprocal Rank Fusion (RRF) formula
            rrf_dense = d_weight / (rrf_k + dense_rank) if dense_rank < 999 else 0.0
            rrf_sparse = sparse_weight / (rrf_k + sparse_rank) if sparse_rank < 999 else 0.0
            rrf_score = rrf_dense + rrf_sparse

            # Best document representation and metadata
            rep = dense_info[0] if dense_info else sparse_info[0]

            scored_candidates.append({
                "product_id": pid,
                "name": rep.get("name", ""),
                "price": rep.get("price", ""),
                "price_amount": rep.get("price_amount", 0.0),
                "document": rep.get("document", ""),
                "distance": rep.get("distance", 0.0),
                "dense_rank": dense_rank if dense_rank < 999 else None,
                "sparse_rank": sparse_rank if sparse_rank < 999 else None,
                "sparse_score": sparse_info[0].get("sparse_score", 0.0) if sparse_info else 0.0,
                "rrf_score": round(rrf_score, 6),
                "strategy": "hybrid",
                "metadata": rep.get("metadata", {}),
            })

        # Sort descending by RRF score
        scored_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored_candidates[:n_results]

    def retrieve_context(
        self,
        query: str,
        n_results: int = 3,
        strategy: str = "hybrid",
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Main retrieval method routing based on desired strategy (hybrid, dense, sparse)."""
        strategy_clean = strategy.lower().strip()
        if strategy_clean == "dense":
            return self.retrieve_dense(query, n_results)
        elif strategy_clean == "sparse":
            return self.retrieve_sparse(query, n_results)
        else:
            return self.retrieve_hybrid(query, n_results, dense_weight=dense_weight)

    def get_status(self) -> Dict[str, Any]:
        collection = self.get_collection()
        if self._bm25.count() == 0:
            self.index_catalog()

        return {
            "status": "connected",
            "mode": self._mode,
            "host": settings.CHROMA_HOST,
            "port": settings.CHROMA_PORT,
            "collection": settings.CHROMA_COLLECTION,
            "document_count": collection.count(),
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL if settings.is_openai_configured else "default_minilm",
            "hybrid_enabled": True,
            "bm25_documents_count": self._bm25.count(),
        }

rag_service = RAGService()
