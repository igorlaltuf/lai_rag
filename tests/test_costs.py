from src.costs import estimate_cost


def test_estimate_cost_for_nano_query():
    cost = estimate_cost("gpt-5.4-nano", input_tokens=10_000, output_tokens=1_200)
    assert round(cost.usd, 4) == 0.0035
