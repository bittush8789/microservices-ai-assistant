import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.schemas import Product, PriceUSD

class ProductCatalog:
    def __init__(self, json_path: Optional[str] = None):
        self.products: List[Product] = []
        self._products_by_id: Dict[str, Product] = {}
        self._products_by_name: Dict[str, Product] = {}
        self._load_catalog(json_path)

    def _locate_products_json(self, explicit_path: Optional[str] = None) -> Path:
        if explicit_path:
            p = Path(explicit_path)
            if p.is_file():
                return p

        current_file = Path(__file__).resolve()
        candidates = [
            current_file.parent / "data" / "products.json",
            current_file.parent.parent.parent / "productcatalogservice" / "products.json",
            Path.cwd() / "src" / "productcatalogservice" / "products.json",
            Path.cwd() / "products.json",
        ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError("Could not find products.json in catalog search paths.")

    def _load_catalog(self, json_path: Optional[str] = None) -> None:
        target_path = self._locate_products_json(json_path)
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_products = data.get("products", [])
        self.products = []
        self._products_by_id = {}
        self._products_by_name = {}

        for item in raw_products:
            price_raw = item.get("priceUsd", {})
            units = price_raw.get("units", 0)
            nanos = price_raw.get("nanos", 0)
            amount = round(units + (nanos / 1_000_000_000.0), 2)
            formatted = f"${amount:.2f}"

            price_obj = PriceUSD(
                currencyCode=price_raw.get("currencyCode", "USD"),
                units=units,
                nanos=nanos,
                amount=amount,
                formatted=formatted,
            )

            product = Product(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                picture=item.get("picture", ""),
                priceUsd=price_obj,
                categories=item.get("categories", []),
            )

            self.products.append(product)
            self._products_by_id[product.id.upper()] = product
            self._products_by_name[product.name.lower()] = product

    def get_all(self) -> List[Product]:
        return self.products

    def get_by_id(self, product_id: str) -> Optional[Product]:
        return self._products_by_id.get(product_id.strip().upper())

    def get_by_name(self, name: str) -> Optional[Product]:
        name_clean = name.strip().lower()
        if name_clean in self._products_by_name:
            return self._products_by_name[name_clean]
        for prod_name, product in self._products_by_name.items():
            if name_clean in prod_name or prod_name in name_clean:
                return product
        return None

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[Product]:
        results = self.products

        if category:
            cat_clean = category.strip().lower()
            results = [p for p in results if any(cat_clean in c.lower() for c in p.categories)]

        if min_price is not None:
            results = [p for p in results if p.price_usd.amount >= min_price]

        if max_price is not None:
            results = [p for p in results if p.price_usd.amount <= max_price]

        if query:
            q_clean = query.strip().lower()
            phrase_matches = [
                p for p in results
                if q_clean in p.name.lower()
                or q_clean in p.description.lower()
                or any(q_clean in c.lower() for c in p.categories)
            ]
            if phrase_matches:
                results = phrase_matches
            else:
                words = [w for w in q_clean.split() if len(w) > 2 and w not in ("the", "are", "how", "much", "for", "what")]
                if words:
                    results = [
                        p for p in results
                        if any(w in p.name.lower() or w in p.description.lower() or any(w in c.lower() for c in p.categories) for w in words)
                    ]

        return results

    def get_catalog_summary_for_prompt(self) -> str:
        lines = []
        for p in self.products:
            cats = ", ".join(p.categories)
            lines.append(
                f"- [{p.id}] {p.name} | Price: {p.price_usd.formatted} | Categories: {cats}\n"
                f"  Description: {p.description}"
            )
        return "\n".join(lines)

catalog = ProductCatalog()
