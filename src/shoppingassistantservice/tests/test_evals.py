import pytest
from evals.evaluator import AssistantEvaluator

@pytest.mark.asyncio
async def test_evaluator_loads_dataset():
    evaluator = AssistantEvaluator()
    assert len(evaluator.test_cases) >= 15
    for tc in evaluator.test_cases:
        assert "id" in tc
        assert "query" in tc

@pytest.mark.asyncio
async def test_evaluator_rag_retrieval_hit_rate():
    evaluator = AssistantEvaluator()
    metrics = await evaluator.evaluate_rag_retrieval()
    assert metrics["hit_rate_at_3"] >= 0.80
    assert metrics["mrr"] > 0.60
    assert metrics["total"] > 0

@pytest.mark.asyncio
async def test_evaluator_guardrail_defense_rate():
    evaluator = AssistantEvaluator()
    metrics = evaluator.evaluate_guardrails()
    assert metrics["defense_rate"] == 1.0
    assert metrics["intercepted"] > 0

@pytest.mark.asyncio
async def test_evaluator_pii_masking():
    evaluator = AssistantEvaluator()
    metrics = evaluator.evaluate_pii_masking()
    assert metrics["mask_rate"] == 1.0

@pytest.mark.asyncio
async def test_evaluator_full_scorecard():
    evaluator = AssistantEvaluator()
    scorecard = await evaluator.run_full_evaluation()
    assert scorecard["status"] == "passed"
    assert scorecard["composite_score"] >= 90.0
    assert "rag_retrieval_hit_rate_at_3" in scorecard["metrics"]
    assert "guardrail_defense_rate" in scorecard["metrics"]
    assert "pricing_accuracy_rate" in scorecard["metrics"]
    assert "pill_generation_rate" in scorecard["metrics"]
