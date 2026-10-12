"""Check parser compatibility with the stock dataset, without generating answers."""
import asyncio
import json
from pathlib import Path

from moonshot.src.connectors.connector_response import ConnectorResponse


def test_every_stock_gold_answer_is_supported_by_configured_metric(load_asset):
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root / "recipes/gsm8k-test.json").read_text(encoding="utf-8"))
    assert len(recipe["datasets"]) == len(recipe["metrics"]) == 1
    data = json.loads((root / "datasets" / (recipe["datasets"][0] + ".json")).read_text(encoding="utf-8"))
    metric = load_asset("metrics/" + recipe["metrics"][0] + ".py").ExactStrMatchGSM8k()
    examples = data["examples"]
    assert examples
    # Synthetic perfect responses test accepted syntax, not model quality or gold truth.
    result = asyncio.run(metric.get_results(
        [row["input"] for row in examples],
        [ConnectorResponse(response=f"The answer is {row['target']}.") for row in examples],
        [row["target"] for row in examples],
    ))["exactstrmatch_gsm8k"]
    assert result["accuracy"] == 100
    assert result["results"] == [1] * len(examples)
    assert len(result["individual_scores"]) == len(examples)
