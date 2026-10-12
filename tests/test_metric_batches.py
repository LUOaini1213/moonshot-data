"""A score must account for every requested sample, including wrong answers."""
import asyncio

import pytest
from moonshot.src.connectors.connector_response import ConnectorResponse


METRICS = [
    ("exactstrmatch", "ExactStrMatch", "A", "A"),
    ("relaxstrmatch", "RelaxStrMatch", "A", "a"),
    ("reverseexactstrmatch", "ReverseExactStrMatch", "A", "B"),
    ("exactstrmatch-f1-squad-v2", "ExactStrMatchF1SquadV2", "Paris", {"text": ["Paris"]}),
]


@pytest.mark.parametrize("stem,cls,prediction,target", METRICS)
@pytest.mark.parametrize("counts", [(2, 1, 2), (1, 2, 1), (1, 2, 2), (2, 2, 1), (0, 0, 0), (1, 0, 1)])
def test_incomplete_batches_are_rejected(load_asset, stem, cls, prediction, target, counts):
    metric = getattr(load_asset(f"metrics/{stem}.py"), cls)()
    prompts = [f"q{i}" for i in range(counts[0])]
    predictions = [ConnectorResponse(response=prediction) for _ in range(counts[1])]
    targets = [target for _ in range(counts[2])]
    with pytest.raises(ValueError, match="non-empty.*same length"):
        asyncio.run(metric.get_results(prompts, predictions, targets))


@pytest.mark.parametrize("stem,cls,prediction,target", METRICS)
def test_full_batch_has_consistent_aggregate_and_per_row_evidence(load_asset, stem, cls, prediction, target):
    metric = getattr(load_asset(f"metrics/{stem}.py"), cls)()
    wrong = "B" if stem == "reverseexactstrmatch" else "wrong answer"
    result = asyncio.run(metric.get_results(
        ["one", "two"], [ConnectorResponse(response=prediction), ConnectorResponse(response=wrong)],
        [target, target],
    ))
    key = "f1" if stem.endswith("squad-v2") else "accuracy"
    assert result["grading_criteria"][key] == 50
    details = next(v for k, v in result.items() if k != "grading_criteria")["individual_scores"]
    rows = details if isinstance(details, list) else details["successful"] + details["unsuccessful"]
    assert len(rows) == 2
    assert {row["prompt"] for row in rows} == {"one", "two"}


@pytest.mark.parametrize("target", ["New\nYork", "New\tYork", ["wrong", "New\nYork"]])
def test_relaxed_matching_normalizes_whitespace_consistently(load_asset, target):
    metric = load_asset("metrics/relaxstrmatch.py").RelaxStrMatch()
    result = asyncio.run(metric.get_results(["city"], [ConnectorResponse(response="New York")], [target]))
    assert result["grading_criteria"]["accuracy"] == 100


def test_exact_and_reverse_matching_keep_multi_target_contract(load_asset):
    for stem, cls, expected in [("exactstrmatch", "ExactStrMatch", 100), ("reverseexactstrmatch", "ReverseExactStrMatch", 0)]:
        metric = getattr(load_asset(f"metrics/{stem}.py"), cls)()
        result = asyncio.run(metric.get_results(["q"], [ConnectorResponse(response="B")], [["A", "B"]]))
        assert result["grading_criteria"]["accuracy"] == expected


@pytest.mark.parametrize("stem,cls,prediction,target", METRICS)
@pytest.mark.parametrize("position,value", [(0, "q"), (2, "1"), (1, None)])
def test_scalar_is_not_silently_treated_as_a_batch(load_asset, stem, cls, prediction, target, position, value):
    metric = getattr(load_asset(f"metrics/{stem}.py"), cls)()
    batches = [["q"], [ConnectorResponse(response=prediction)], [target]]
    batches[position] = value
    with pytest.raises(ValueError, match="non-empty.*same length"):
        asyncio.run(metric.get_results(*batches))
