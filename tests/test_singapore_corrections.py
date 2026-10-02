"""Protect the option/key corrections submitted in upstream PR #223.

These are consistency regressions, not a claim to fact-check every dataset item.
"""
import json
from pathlib import Path
import re

import pytest

DATA = Path(__file__).resolve().parents[1] / "datasets"


@pytest.mark.parametrize("dataset,index,expected", [
    ("singapore-transport-system", 15, "A) 1987"),
    ("singapore-transport-system", 26, "B) 2001"),
    ("singapore-public-housing", 2, "A) 5 years"),
])
def test_corrected_key_matches_one_complete_option(dataset, index, expected):
    document = json.loads((DATA / f"{dataset}.json").read_text(encoding="utf-8"))
    # The upstream issue uses zero-based example indices.
    example = document["examples"][index]
    options = re.findall(r"^[A-D]\) .+$", example["input"], re.MULTILINE)
    assert len(options) == 4
    assert example["target"] == expected
    assert options.count(example["target"]) == 1
    if dataset == "singapore-public-housing":
        assert "Standard HDB flat" in example["input"]


@pytest.mark.parametrize("dataset,expected", [
    ("singapore-public-housing", "Singapore Public Housing"),
    ("singapore-political-history", "Singapore Political History"),
    ("singapore-pofma-statements-2023", "Singapore POFMA Statements"),
    ("singapore-pofma-statements-2024", "Singapore POFMA Statements"),
    ("singapore-pofma-true-statements", "Singapore POFMA True Statements"),
])
def test_corrected_metadata_names(dataset, expected):
    document = json.loads((DATA / f"{dataset}.json").read_text(encoding="utf-8"))
    assert document["name"] == expected
    assert document["examples"]
