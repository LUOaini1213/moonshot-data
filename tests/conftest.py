"""Offline asset tests against the real Moonshot framework interfaces."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

os.environ["MS_LOG_TO_FILE"] = "false"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def load_asset(monkeypatch):
    # Keep configuration IO out of metric tests; scoring code and interfaces are real.
    from moonshot.src.metrics.metric_interface import MetricInterface

    monkeypatch.setattr(MetricInterface, "get_metrics_configuration", lambda self, name: {})

    def load(relative):
        path = ROOT / relative
        name = "offline_asset_" + relative.replace("/", "_").replace("-", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return load
