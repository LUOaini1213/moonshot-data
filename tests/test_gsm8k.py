"""GSM8K scoring through the actual Moonshot 0.7.6 response/interface types."""
import asyncio
from decimal import localcontext

import pytest
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.metrics.metric_interface import MetricInterface


@pytest.fixture
def metric_module(load_asset):
    return load_asset("metrics/exactstrmatch-gsm8k-test.py")


@pytest.fixture
def metric(metric_module):
    value = metric_module.ExactStrMatchGSM8k()
    assert isinstance(value, MetricInterface)
    return value


def score(metric, predictions, targets, prompts=None):
    responses = [ConnectorResponse(response=text) for text in predictions]
    if prompts is None:
        prompts = [f"question-{i}" for i in range(len(responses))]
    return asyncio.run(metric.get_results(prompts, responses, targets))


@pytest.mark.parametrize("predicted,target", [
    ("112", "12"), ("-12", "12"), ("123456", "123457"),
    ("9007199254740993", "9007199254740992"),
    ("0.123456789012345678901234567891", "0.123456789012345678901234567890"),
    ("1e3", "1"), ("1e-3", "1"),
])
def test_different_complete_numbers_never_match(metric, predicted, target):
    result = score(metric, [f"The answer is {predicted}."], [target])["exactstrmatch_gsm8k"]
    assert result["results"] == [0]
    assert result["accuracy"] == 0
    assert result["individual_scores"][0]["matched"] is False


@pytest.mark.parametrize("predicted,target,extracted", [
    ("42", "42", "42"), ("$1,234.50", "1234.5", "1234.50"),
    ("-1,234.50", "-1234.5", "-1234.50"), ("+12", "12", "+12"),
    (".5", "0.50", ".5"), ("-.5", "-0.5", "-.5"),
    ("1e3", "1000", "1e3"), ("-1.25E+3", "-1250", "-1.25E+3"),
    ("1.25e-3", "0.00125", "1.25e-3"), ("-0.0", "0", "-0.0"),
    ("1,234,567", "1234567", "1234567"),
])
def test_supported_numeric_formats_have_exact_value_and_evidence(metric, predicted, target, extracted):
    result = score(metric, [f"The answer is {predicted}."], [target])
    assert result["exactstrmatch_gsm8k"] == {
        "accuracy": 100.0,
        "results": [1],
        "individual_scores": [{"prompt": "question-0", "predicted_value": extracted,
                               "target": target, "matched": True}],
    }
    assert result["grading_criteria"] == {"accuracy": 100.0}


@pytest.mark.parametrize("token", [
    ".", ",", "..", "1.2.3", "1e", "1e+", "1e-", "1e3kg", "12kg",
    "1,23", "12,34", "1,2345", "1,234,", "--1", "NaN", "Infinity", "-inf", "1/2",
])
def test_invalid_numeric_tokens_are_unmatched_without_truncation_or_crash(metric, token):
    result = score(metric, [f"The answer is {token}"], ["1"])["exactstrmatch_gsm8k"]
    assert result["results"] == [0]
    assert result["individual_scores"][0]["predicted_value"] == "NA"


@pytest.mark.parametrize("response,expected", [
    ("Reasoning: 1 + 2 = 3. The answer is 3.", "3"),
    ("The answer is 1. Revised: The answer is 3.", "3"),
    ("The answer is 1. Revised: The answer is .", "NA"),
    ("The answer is 1. Revised: The answer is", "NA"),
    ("The answer is 3 apples.", "3"),
    ("The answer is.5", "NA"), ("The answer is+3", "NA"),
    ("3", "NA"), ("No answer available.", "NA"), ("", "NA"),
])
def test_explicit_template_and_last_answer_contract(metric, response, expected):
    assert metric.extract(response) == expected


@pytest.mark.parametrize("location", ["begin", "end", "any", "exact"])
def test_numeric_matching_is_whole_value_even_when_text_location_is_requested(metric_module, location):
    assert metric_module.inspect_match_str("112", "12", location=location, numeric=True)[1] is False
    assert metric_module.inspect_match_str("1e3", "1000", location=location, numeric=True)[1] is True
    assert metric_module.inspect_match_str("1.2.3", "1.2", location=location, numeric=True)[1] is False


@pytest.mark.parametrize("target", ["NaN", "Infinity", "NA", "1.2.3", "1,23", ""])
def test_invalid_targets_cannot_match(metric, target):
    assert score(metric, [f"The answer is {target}"], [target])["exactstrmatch_gsm8k"]["results"] == [0]


def test_aggregate_and_every_individual_score_stay_aligned(metric):
    result = score(metric, ["The answer is 12.", "The answer is 112.", "The answer is .", "The answer is 1e3."],
                   [12, "12", "0", "1000"], prompts=["a", "b", "c", "d"])
    assert result["exactstrmatch_gsm8k"] == {
        "accuracy": 50.0, "results": [1, 0, 0, 1],
        "individual_scores": [
            {"prompt": "a", "predicted_value": "12", "target": "12", "matched": True},
            {"prompt": "b", "predicted_value": "112", "target": "12", "matched": False},
            {"prompt": "c", "predicted_value": "NA", "target": "0", "matched": False},
            {"prompt": "d", "predicted_value": "1e3", "target": "1000", "matched": True},
        ],
    }
    assert result["grading_criteria"] == {"accuracy": 50.0}


@pytest.mark.parametrize("counts", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 2, 2), (2, 1, 2), (2, 2, 1)])
def test_empty_or_unequal_batches_are_rejected(metric, counts):
    n_prompts, n_predictions, n_targets = counts
    with pytest.raises(ValueError, match="non-empty|same length"):
        score(metric, ["The answer is 1."] * n_predictions, ["1"] * n_targets, ["q"] * n_prompts)


@pytest.mark.parametrize("batch_index,invalid", [(0, None), (1, None), (2, None), (0, "q"), (2, "1")])
def test_invalid_batch_containers_raise_clear_value_errors(metric, batch_index, invalid):
    batches = [["q"], [ConnectorResponse(response="The answer is 1.")], ["1"]]
    batches[batch_index] = invalid
    with pytest.raises(ValueError, match="non-empty.*same length"):
        asyncio.run(metric.get_results(*batches))


def test_decimal_context_does_not_round_scoring_or_normalization(metric, metric_module):
    number = "123456789.12345678901234567890123456789"
    with localcontext() as context:
        context.prec = 3
        result = score(metric, [f"The answer is {number}."], [number[:-1] + "8"])
        assert result["exactstrmatch_gsm8k"]["results"] == [0]
        assert metric_module.normalize_number(number) == number
