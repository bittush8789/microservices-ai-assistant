import json
import re
from typing import List, Optional, Dict, Any, Tuple
from openai import OpenAI
from app.config import settings
from app.catalog import catalog
from app.rag import rag_service
from app.guardrails import guardrail_manager
from app.schemas import ChatMessage, ChatResponse, Product

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product including description, categories, and price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id_or_name": {
                        "type": "string",
                        "description": "The ID (e.g., OLJCESPC7Z) or name (e.g., Sunglasses) of the product."
                    }
                },
                "required": ["product_id_or_name"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_pricing",
            "description": "Get accurate current pricing and currency details for one or more products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id_or_name": {
                        "type": "string",
                        "description": "The product ID or product name to get price for."
                    }
                },
                "required": ["product_id_or_name"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog by keyword, category, and price range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or term (e.g., 'summer', 'sunglasses', 'gift', 'kitchen')."
                    },
                    "category": {
                        "type": "string",
                        "description": "Category filter (e.g., 'accessories', 'clothing', 'kitchen', 'decor', 'footwear', 'hair', 'beauty')."
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in USD."
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price in USD."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Perform Hybrid RAG search (combining dense vector embeddings and sparse BM25 lexical matching) for internal product specifications, materials, warranty, and detailed features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or question regarding product features, materials, care, or specifications."
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["hybrid", "dense", "sparse"],
                        "description": "Retrieval strategy: 'hybrid' (default), 'dense' (Pinecone), or 'sparse' (BM25)."
                    }
                },
                "required": ["query"],
            }
        }
    }
]

def _execute_tool(tool_name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json)
    except Exception as e:
        return json.dumps({"error": f"Invalid arguments JSON: {e}"})

    if tool_name in ("get_product_details", "get_product_pricing"):
        query = args.get("product_id_or_name", "")
        prod = catalog.get_by_id(query) or catalog.get_by_name(query)
        if not prod:
            return json.dumps({"error": f"Product '{query}' not found in catalog."})

        if tool_name == "get_product_pricing":
            return json.dumps({
                "id": prod.id,
                "name": prod.name,
                "price": prod.price_usd.formatted,
                "amount": prod.price_usd.amount,
                "currency": prod.price_usd.currency_code
            })
        else:
            return json.dumps(prod.model_dump())

    elif tool_name == "search_products":
        results = catalog.search(
            query=args.get("query"),
            category=args.get("category"),
            min_price=args.get("min_price"),
            max_price=args.get("max_price"),
        )
        return json.dumps([p.model_dump() for p in results])

    elif tool_name == "search_knowledge_base":
        results = rag_service.retrieve_context(
            query=args.get("query", ""),
            n_results=3,
            strategy=args.get("strategy", "hybrid"),
        )
        return json.dumps(results)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})

def extract_ids_from_text(text: str) -> List[str]:
    pattern = r"\[([a-zA-Z0-9-]+)\]"
    matches = re.findall(pattern, text)
    valid_ids = []
    for m in matches:
        if catalog.get_by_id(m) and m not in valid_ids:
            valid_ids.append(m)
    return valid_ids

def generate_follow_up_pills(query: str, products: List[Product], content: str) -> List[str]:
    """Generates dynamic, contextual follow-up suggestion pills based on user inquiry and retrieved products."""
    pills: List[str] = []
    q_lower = query.lower()

    if any(k in q_lower for k in ["sunglass", "glasses", "shades"]):
        pills.extend(["Check sunglasses price", "Are they polarized?", "Accessories under $25"])
    elif any(k in q_lower for k in ["watch", "time", "clock"]):
        pills.extend(["Is the watch water resistant?", "What is the warranty?", "Explore accessories"])
    elif any(k in q_lower for k in ["hair", "dryer", "blow"]):
        pills.extend(["Travel hairdryer voltage", "Hairdryer warranty", "Salon styling tools"])
    elif any(k in q_lower for k in ["kitchen", "jar", "candle", "shaker", "mug"]):
        pills.extend(["Kitchen items under $20", "Bamboo glass jar dimensions", "Explore home decor"])
    elif any(k in q_lower for k in ["clothing", "top", "tank", "shirt", "apparel"]):
        pills.extend(["Tank top fabric material", "Washing instructions", "Footwear & loafers"])
    elif any(k in q_lower for k in ["price", "cost", "cheap", "expensive", "under", "dollar", "$"]):
        pills.extend(["Top products under $20", "Show sunglasses", "Check watch price"])
    else:
        if products:
            p = products[0]
            pills.append(f"More about {p.name}")
            pills.append(f"What is {p.name} made of?")
            pills.append("Items in same category")
        else:
            pills = ["Show sunglasses", "Watch features & price", "Kitchenware under $20", "Travel hairdryer"]

    # Filter duplicates and limit to 4
    unique_pills = []
    for pill in pills:
        if pill not in unique_pills:
            unique_pills.append(pill)
    return unique_pills[:4]

class AIAssistant:
    def __init__(self):
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> Optional[OpenAI]:
        if not settings.is_openai_configured:
            return None
        if self._client is None:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def _build_system_prompt(self, rag_context: Optional[str] = None) -> str:
        catalog_summary = catalog.get_catalog_summary_for_prompt()
        rag_section = ""
        if rag_context:
            rag_section = f"""
Internal Product Knowledge Retrieved from Pinecone Vector DB:
---------------------------------------------------
{rag_context}
---------------------------------------------------
"""

        return f"""You are the friendly, knowledgeable AI Shopping Assistant for Online Boutique.
Your role is to assist customers with:
1. Product-related information (descriptions, features, styles, categories, recommendations).
2. Accurate pricing information (current prices in USD, comparisons, budget recommendations).
3. Internal specifications, materials, warranty, care instructions, and dimensions.

Here is the Official Product Catalog with exact prices:
{catalog_summary}
{rag_section}
Guidelines for your responses:
- Always provide accurate pricing based on the catalog and retrieved records above. Never invent or guess prices.
- When recommending or discussing a product, include its product ID in brackets like [PRODUCT_ID] (e.g. [OLJCESPC7Z] for Sunglasses) so our system can display product cards.
- If a customer asks for internal product specifications (materials, warranty, dimensions, travel suitability, etc.), use the Pinecone Vector DB retrieved knowledge.
- If a customer asks for items under a certain budget (e.g. under $20), recommend items matching that price range with their exact prices.
- Be polite, helpful, concise, and enthusiastic about helping customers find what they need.
"""

    def _local_fallback_response(self, user_message: str, rag_results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        msg = user_message.lower()

        # Gather IDs from RAG results and catalog search
        rag_ids = [r["product_id"] for r in rag_results if r.get("product_id")]
        catalog_results = catalog.search(query=user_message)
        catalog_ids = [p.id for p in catalog_results]

        combined_ids = []
        for pid in rag_ids + catalog_ids:
            if pid not in combined_ids:
                combined_ids.append(pid)

        if not combined_ids:
            if "price" in msg or "cost" in msg or "how much" in msg:
                content = (
                    "Here is the pricing information for our catalog:\n"
                    + "\n".join(f"- [{p.id}] **{p.name}**: {p.price_usd.formatted} ({p.description})" for p in catalog.get_all())
                    + "\n\n*(Note: Running in offline catalog mode because OPENAI_API_KEY is not yet configured in .env.openai)*"
                )
                return content, [p.id for p in catalog.get_all()]

            content = (
                "Welcome to Online Boutique! I can help you with product information and pricing.\n"
                "Here are some of our popular products:\n"
                + "\n".join(f"- [{p.id}] **{p.name}** - {p.price_usd.formatted}: {p.description}" for p in catalog.get_all()[:4])
                + "\n\n*(Note: Please set OPENAI_API_KEY in .env.openai to enable full natural language conversational AI)*"
            )
            return content, [p.id for p in catalog.get_all()[:4]]

        content = "Here are the relevant products retrieved from our Pinecone Vector DB product knowledge base:\n"
        for pid in combined_ids:
            p = catalog.get_by_id(pid)
            if p:
                content += f"- [{p.id}] **{p.name}** ({p.price_usd.formatted}): {p.description}\n"

        if rag_results:
            top_rag = rag_results[0]
            content += f"\n**Highlighted Specifications for [{top_rag.get('product_id')}]:**\n"
            # Add short snippet from document
            doc_lines = top_rag.get("document", "").split("\n")
            spec_lines = [l for l in doc_lines if l.startswith("- ")][:3]
            if spec_lines:
                content += "\n".join(spec_lines) + "\n"

        content += "\n*(Note: Please configure OPENAI_API_KEY in .env.openai for full AI conversational responses)*"
        return content, combined_ids

    async def chat(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        image: Optional[str] = None,
    ) -> ChatResponse:
        # Step 0: Input Guardrails Check
        guardrail_res = guardrail_manager.validate_input(message)
        if not guardrail_res.is_safe:
            return ChatResponse(
                content=guardrail_res.intercept_response or "Request intercepted by security policy.",
                products=[],
                extracted_ids=[],
                pills=guardrail_res.suggested_pills or ["Show sunglasses", "Watch specifications", "Kitchenware under $20"],
                guardrails={
                    "status": "intercepted",
                    "action": guardrail_res.action,
                    "reason": guardrail_res.reason,
                    "metadata": guardrail_res.metadata,
                },
                details={
                    "mode": "guardrail_intercepted",
                    "reason": guardrail_res.reason,
                },
            )

        sanitized_query = guardrail_res.sanitized_message or message

        # Step 1: Retrieve context from Pinecone Vector DB via Hybrid RAG
        rag_results = rag_service.retrieve_context(query=sanitized_query, n_results=3)
        rag_context_text = "\n\n".join(
            f"Result {i+1} (Product {r.get('product_id')} - {r.get('name')}, Price: {r.get('price')}):\n{r.get('document')}"
            for i, r in enumerate(rag_results)
        )

        client = self._get_client()

        # If OpenAI is not configured, use local RAG-powered fallback
        if client is None:
            content, extracted_ids = self._local_fallback_response(sanitized_query, rag_results)
            products = [catalog.get_by_id(pid) for pid in extracted_ids if catalog.get_by_id(pid)]

            # Output Guardrails verification
            verified_content, out_meta = guardrail_manager.verify_output(content, products)
            pills = generate_follow_up_pills(sanitized_query, products, verified_content)

            return ChatResponse(
                content=verified_content,
                products=products,
                extracted_ids=extracted_ids,
                pills=pills,
                guardrails={
                    "status": "passed",
                    "input_action": guardrail_res.action,
                    "output_verification": out_meta,
                },
                details={
                    "mode": "fallback_rag",
                    "rag_mode": rag_service._mode,
                    "rag_results_count": len(rag_results),
                    "reason": "OPENAI_API_KEY not configured",
                },
            )

        system_prompt = self._build_system_prompt(rag_context=rag_context_text)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if history:
            for h in history:
                messages.append({"role": h.role, "content": h.content})

        user_content: Any = sanitized_query
        if image:
            user_content = [
                {"type": "text", "text": sanitized_query},
                {"type": "image_url", "image_url": {"url": image}},
            ]

        messages.append({"role": "user", "content": user_content})

        try:
            # Initial call with tools
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Execute tool calls if requested by model
            if tool_calls:
                messages.append(response_message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    tool_result = _execute_tool(function_name, function_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result,
                    })

                # Second call to generate final answer after tools
                second_response = client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                    temperature=0.7,
                )
                final_content = second_response.choices[0].message.content or ""
            else:
                final_content = response_message.content or ""

            extracted_ids = extract_ids_from_text(final_content)
            # Also include IDs from RAG results if not already present
            for r in rag_results:
                pid = r.get("product_id")
                if pid and pid in final_content and pid not in extracted_ids:
                    extracted_ids.append(pid)

            products = [catalog.get_by_id(pid) for pid in extracted_ids if catalog.get_by_id(pid)]

            # Output Guardrails verification
            verified_content, out_meta = guardrail_manager.verify_output(final_content, products)
            pills = generate_follow_up_pills(sanitized_query, products, verified_content)

            return ChatResponse(
                content=verified_content,
                products=products,
                extracted_ids=extracted_ids,
                pills=pills,
                guardrails={
                    "status": "passed",
                    "input_action": guardrail_res.action,
                    "output_verification": out_meta,
                },
                details={
                    "model": settings.OPENAI_MODEL,
                    "tools_used": bool(tool_calls),
                    "rag_mode": rag_service._mode,
                    "rag_results_count": len(rag_results),
                },
            )

        except Exception as e:
            content, fallback_ids = self._local_fallback_response(sanitized_query, rag_results)
            products = [catalog.get_by_id(pid) for pid in fallback_ids if catalog.get_by_id(pid)]
            verified_content, out_meta = guardrail_manager.verify_output(content, products)
            pills = generate_follow_up_pills(sanitized_query, products, verified_content)

            return ChatResponse(
                content=f"Error connecting to OpenAI ({str(e)}).\n\n{verified_content}",
                products=products,
                extracted_ids=fallback_ids,
                pills=pills,
                guardrails={
                    "status": "passed",
                    "input_action": guardrail_res.action,
                    "output_verification": out_meta,
                },
                details={"error": str(e), "mode": "error_fallback", "rag_mode": rag_service._mode},
            )

assistant = AIAssistant()
