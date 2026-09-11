import json
import re
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
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

SEMANTIC_CLUSTERS = {
    "sunglasses": ["eyewear", "shades", "eye", "protection", "sunny", "sun", "polarized", "glare", "aviator", "uv400", "beach", "sunglass"],
    "watch": ["timepiece", "time", "clock", "wrist", "quartz", "horology", "gold", "watch"],
    "hairdryer": ["hair", "dryer", "blowdryer", "blow", "drying", "salon", "styling", "voltage", "hairdryer"],
    "loafers": ["shoes", "footwear", "loafer", "shoe", "feet", "leather", "slipon"],
    "tanktop": ["tank", "top", "shirt", "apparel", "clothing", "sleeveless", "crop", "cotton"],
    "candleholder": ["candle", "holder", "votive", "tealight", "decor", "ambiance", "ceramic", "iron"],
    "mug": ["cup", "drinkware", "coffee", "tea", "beverage", "mug", "stoneware"],
    "jar": ["canister", "container", "storage", "pasta", "beans", "jar", "bamboo"],
    "shakers": ["seasoning", "salt", "pepper", "spices", "condiments", "shaker"],
}

def compute_fallback_embedding(text: str, dimension: int = 1536) -> List[float]:
    """
    Deterministic semantic hash embedding generator for local testing and offline fallback.
    Produces an L2-normalized float vector with domain semantic expansion.
    """
    text_lower = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+", text_lower)
    expanded = list(tokens)

    for cluster_name, syns in SEMANTIC_CLUSTERS.items():
        if any(s in text_lower for s in syns):
            expanded.extend(syns[:4])
            expanded.append(cluster_name)

    vec = [0.0] * dimension
    for i, token in enumerate(expanded):
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dimension
        weight = 1.0 / (1.0 + 0.05 * i)
        vec[idx] += weight

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec[0] = 1.0
    return vec

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

class RAGService:
    """
    Hybrid RAG Engine:
    - Dense Vector Retrieval: Powered by Pinecone Serverless Index (with local fallback).
    - Sparse Lexical Retrieval: Powered by BM25Okapi.
    - Fusion: Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self._pinecone_client = None
        self._index = None
        self._mode: str = "uninitialized"
        self._bm25 = BM25Index()

        # In-memory vector store for fallback/test mode
        self._mock_vectors: Dict[str, List[float]] = {}
        self._mock_metadatas: Dict[str, Dict[str, Any]] = {}
        self._mock_documents: Dict[str, str] = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding via OpenAI if configured, or uses deterministic fallback embedding."""
        if settings.is_openai_configured:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                resp = client.embeddings.create(
                    input=text,
                    model=settings.OPENAI_EMBEDDING_MODEL,
                )
                return resp.data[0].embedding
            except Exception as e:
                logger.warning(f"OpenAI embedding generation failed ({e}), using fallback embedding.")

        return compute_fallback_embedding(text, dimension=settings.PINECONE_DIMENSION)

    def _init_pinecone(self):
        """Initializes connection to Pinecone Vector Database or falls back to local vector store."""
        if settings.is_pinecone_configured:
            try:
                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                self._pinecone_client = pc

                # Check if index exists or create serverless index
                existing_indexes = [i.name for i in pc.list_indexes()]
                if settings.PINECONE_INDEX_NAME not in existing_indexes:
                    logger.info(f"Creating serverless Pinecone index '{settings.PINECONE_INDEX_NAME}'...")
                    pc.create_index(
                        name=settings.PINECONE_INDEX_NAME,
                        dimension=settings.PINECONE_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region=settings.PINECONE_ENVIRONMENT,
                        ),
                    )

                self._index = pc.Index(settings.PINECONE_INDEX_NAME)
                self._mode = "pinecone_serverless"
                logger.info(f"Connected to Pinecone Serverless Index: {settings.PINECONE_INDEX_NAME}")
                return
            except Exception as e:
                logger.warning(f"Could not connect to Pinecone cloud ({e}). Using local Pinecone fallback engine.")

        # Fallback local vector store mode
        self._mode = "pinecone_mock_local"
        logger.info("Running Pinecone Vector DB in local in-memory fallback mode.")

    def get_index(self):
        if self._mode == "uninitialized":
            self._init_pinecone()
        return self._index

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
        """Indexes the product catalog into Pinecone Vector DB and BM25."""
        if self._mode == "uninitialized":
            self._init_pinecone()

        products = catalog.get_all()
        ids = []
        documents = []
        metadatas = []
        vectors = []

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
                "document": doc_text,
            }
            emb = self._get_embedding(doc_text)

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(meta)
            vectors.append(emb)

        # Build BM25 sparse index
        self._bm25.build(ids=ids, documents=documents, metadatas=metadatas)

        # Index into Pinecone (Cloud or Local Fallback)
        if self._mode == "pinecone_serverless" and self._index is not None:
            try:
                upsert_data = [
                    {"id": doc_id, "values": vec, "metadata": meta}
                    for doc_id, vec, meta in zip(ids, vectors, metadatas)
                ]
                self._index.upsert(vectors=upsert_data, namespace=settings.PINECONE_NAMESPACE)
                logger.info(f"Successfully upserted {len(ids)} vectors into Pinecone Serverless Index.")
            except Exception as e:
                logger.error(f"Failed to upsert to Pinecone: {e}. Falling back to local index.")
                self._mode = "pinecone_mock_local"

        # Local fallback store
        for doc_id, vec, meta, doc in zip(ids, vectors, metadatas, documents):
            self._mock_vectors[doc_id] = vec
            self._mock_metadatas[doc_id] = meta
            self._mock_documents[doc_id] = doc

        logger.info(f"Hybrid RAG: {len(ids)} documents indexed in Pinecone ({self._mode}) and BM25.")
        return len(ids)

    def retrieve_dense(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Dense semantic retrieval via Pinecone."""
        if self._mode == "uninitialized" or len(self._mock_vectors) == 0:
            self.index_catalog()

        query_emb = self._get_embedding(query)
        retrieved: List[Dict[str, Any]] = []

        if self._mode == "pinecone_serverless" and self._index is not None:
            try:
                res = self._index.query(
                    vector=query_emb,
                    top_k=n_results,
                    include_metadata=True,
                    namespace=settings.PINECONE_NAMESPACE,
                )
                for rank, match in enumerate(res.matches, start=1):
                    meta = match.metadata or {}
                    retrieved.append({
                        "product_id": meta.get("product_id", ""),
                        "name": meta.get("name", ""),
                        "price": meta.get("price", ""),
                        "price_amount": float(meta.get("price_amount", 0.0)),
                        "document": meta.get("document", ""),
                        "distance": 1.0 - float(match.score or 0.0),
                        "dense_rank": rank,
                        "strategy": "dense",
                        "metadata": meta,
                    })
                return retrieved
            except Exception as e:
                logger.warning(f"Pinecone query error ({e}), falling back to local dense retrieval.")

        # Local dense retrieval via cosine similarity
        scored_items = []
        for doc_id, vec in self._mock_vectors.items():
            sim = cosine_similarity(query_emb, vec)
            meta = self._mock_metadatas.get(doc_id, {})
            scored_items.append((sim, doc_id, meta))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        for rank, (sim, doc_id, meta) in enumerate(scored_items[:n_results], start=1):
            retrieved.append({
                "product_id": meta.get("product_id", ""),
                "name": meta.get("name", ""),
                "price": meta.get("price", ""),
                "price_amount": float(meta.get("price_amount", 0.0)),
                "document": self._mock_documents.get(doc_id, ""),
                "distance": 1.0 - sim,
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
        Hybrid Retrieval combining Pinecone (Dense) and BM25 (Sparse)
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
        """Main retrieval router supporting 'hybrid', 'dense', or 'sparse'."""
        strategy_clean = strategy.lower().strip()
        if strategy_clean == "dense":
            return self.retrieve_dense(query, n_results)
        elif strategy_clean == "sparse":
            return self.retrieve_sparse(query, n_results)
        else:
            return self.retrieve_hybrid(query, n_results, dense_weight=dense_weight)

    def get_status(self) -> Dict[str, Any]:
        if self._bm25.count() == 0:
            self.index_catalog()

        doc_count = len(self._mock_vectors)
        return {
            "status": "connected",
            "provider": "pinecone",
            "mode": self._mode,
            "index_name": settings.PINECONE_INDEX_NAME,
            "environment": settings.PINECONE_ENVIRONMENT,
            "namespace": settings.PINECONE_NAMESPACE,
            "dimension": settings.PINECONE_DIMENSION,
            "document_count": doc_count,
            "embedding_model": settings.OPENAI_EMBEDDING_MODEL if settings.is_openai_configured else "fallback_semantic_hash",
            "hybrid_enabled": True,
            "bm25_documents_count": self._bm25.count(),
        }

rag_service = RAGService()
