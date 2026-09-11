import pytest
import sys
from pathlib import Path

# Add shoppingassistantservice to sys.path
service_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(service_dir))

from app.catalog import ProductCatalog

@pytest.fixture
def catalog():
    return ProductCatalog()

def test_catalog_loads_products(catalog):
    products = catalog.get_all()
    assert len(products) == 9
    ids = [p.id for p in products]
    assert "OLJCESPC7Z" in ids # Sunglasses
    assert "1YMWWN1N4O" in ids # Watch

def test_pricing_calculation(catalog):
    sunglasses = catalog.get_by_id("OLJCESPC7Z")
    assert sunglasses is not None
    assert sunglasses.name == "Sunglasses"
    assert sunglasses.price_usd.amount == 19.99
    assert sunglasses.price_usd.formatted == "$19.99"
    assert sunglasses.price_usd.currency_code == "USD"

    watch = catalog.get_by_id("1YMWWN1N4O")
    assert watch is not None
    assert watch.price_usd.amount == 109.99
    assert watch.price_usd.formatted == "$109.99"

def test_get_by_name(catalog):
    mug = catalog.get_by_name("Mug")
    assert mug is not None
    assert mug.id == "6E92ZMYYFZ"
    assert mug.price_usd.amount == 8.99

    # Case insensitive partial match
    jar = catalog.get_by_name("glass jar")
    assert jar is not None
    assert jar.name == "Bamboo Glass Jar"

def test_search_by_category(catalog):
    kitchen_items = catalog.search(category="kitchen")
    assert len(kitchen_items) == 3
    names = [p.name for p in kitchen_items]
    assert "Mug" in names
    assert "Salt & Pepper Shakers" in names
    assert "Bamboo Glass Jar" in names

def test_search_by_price_range(catalog):
    cheap_items = catalog.search(max_price=20.0)
    for p in cheap_items:
        assert p.price_usd.amount <= 20.0

    expensive_items = catalog.search(min_price=50.0)
    names = [p.name for p in expensive_items]
    assert "Watch" in names # $109.99
    assert "Loafers" in names # $89.99

def test_catalog_prompt_summary(catalog):
    summary = catalog.get_catalog_summary_for_prompt()
    assert "Sunglasses" in summary
    assert "$19.99" in summary
    assert "[OLJCESPC7Z]" in summary
