"""Offline cache contracts using real Moonshot, its SQLite asset, and a fake model.

Only asset/config discovery and the model transport are replaced. The public
Runner/Run, prompt pipeline, cache SQL, metric inputs, and result formatter run.
"""
import asyncio
import copy
from types import SimpleNamespace

import pytest


@pytest.fixture
def cache(load_asset, tmp_path):
    from moonshot.src.connectors.connector import Connector
    from moonshot.src.connectors.connector_response import ConnectorResponse
    from moonshot.src.runners.runner import Runner

    module = load_asset("runners-modules/benchmarking.py")
    sqlite = load_asset("databases-modules/sqlite.py")
    db = sqlite.SQLite(str(tmp_path / "runner.db"))
    assert db.create_connection()
    db.create_table(Runner.sql_create_runner_cache_table)

    class CountedConnector(Connector):
        def __init__(self):
            self.id = "offline-endpoint"
            self.system_prompt = ""
            self.calls = []

        async def get_response(self, prompt):
            self.calls.append((self.system_prompt, prompt))
            return ConnectorResponse(
                response=f"{self.system_prompt}|{prompt}", context=["offline evidence"]
            )

    connector = CountedConnector()
    errors = []
    benchmark = module.Benchmarking()
    benchmark.database_instance = db
    benchmark.run_progress = SimpleNamespace(notify_error=errors.append)
    state = SimpleNamespace(
        module=module, db=db, connector=connector, benchmark=benchmark, errors=errors
    )
    yield state
    db.close_connection()


def prompt(cache, *, text="question", target="old", index=0, seed=0,
           attack="", system=""):
    from moonshot.src.connectors.connector_prompt_arguments import ConnectorPromptArguments

    return cache.module.PromptArguments(
        rec_id="recipe", ds_id="dataset", pt_id="no-template", random_seed=seed,
        system_prompt=system, attack_module_id=attack,
        connector_prompt=ConnectorPromptArguments(
            prompt_index=index, prompt=text, target=target
        ),
    )


def predict(cache, item):
    cache.connector.set_system_prompt(item.system_prompt)
    result = asyncio.run(cache.benchmark._process_single_prompt(
        item, cache.connector, asyncio.Event()
    ))
    assert cache.errors == []
    assert result is not None
    return result


def test_identical_request_reuses_response_context_and_duration(cache):
    original = prompt(cache, target={"labels": ["A"]}, system="answer briefly")
    first = predict(cache, original)
    second = predict(cache, original)
    assert len(cache.connector.calls) == 1
    assert second.connector_prompt.predicted_results == first.connector_prompt.predicted_results
    assert second.connector_prompt.duration == first.connector_prompt.duration
    assert second.connector_prompt.predicted_results.context == ["offline evidence"]
    assert original.connector_prompt.predicted_results is None


@pytest.mark.parametrize("target", ["1", "True", ["A", "B"], {"label": "B"}, None])
def test_cache_hit_keeps_current_target_type_and_sample_metadata(cache, target):
    predict(cache, prompt(cache, target="obsolete", index=2, seed=1, attack="old"))
    current = prompt(cache, target=target, index=9, seed=42, attack="current")
    saved_input = copy.deepcopy(current)
    result = predict(cache, current)
    assert len(cache.connector.calls) == 1
    assert result.connector_prompt.target == target
    assert type(result.connector_prompt.target) is type(target)
    assert result.connector_prompt.prompt_index == 9
    assert result.random_seed == 42
    assert result.attack_module_id == "current"
    assert current == saved_input


def test_unchanged_string_label_is_not_deserialized_as_number(cache):
    item = prompt(cache, target="1")
    predict(cache, item)
    result = predict(cache, item)
    assert result.connector_prompt.target == "1"
    assert isinstance(result.connector_prompt.target, str)
    assert len(cache.connector.calls) == 1


@pytest.mark.parametrize("system,text", [("new system", "question"), ("", "different question")])
def test_model_request_changes_miss_cache(cache, system, text):
    predict(cache, prompt(cache))
    result = predict(cache, prompt(cache, system=system, text=text))
    assert cache.connector.calls == [("", "question"), (system, text)]
    assert result.connector_prompt.predicted_results.response == f"{system}|{text}"
    assert result.system_prompt == system


@pytest.mark.parametrize("system,expected_calls", [("", 1), ("new system", 2)])
def test_legacy_null_system_prompt_matches_only_empty_system(cache, system, expected_calls):
    first = predict(cache, prompt(cache))
    cache.db.sqlite_conn.execute("UPDATE runner_cache_table SET system_prompt = NULL")
    cache.db.sqlite_conn.commit()
    second = predict(cache, prompt(cache, system=system))
    assert len(cache.connector.calls) == expected_calls
    assert second.system_prompt == system
    assert second.connector_prompt.predicted_results.response == f"{system}|question"
    if not system:
        assert second.connector_prompt.duration == first.connector_prompt.duration


@pytest.fixture
def public_runner(cache, load_asset, monkeypatch, tmp_path):
    from moonshot.src.runners.runner import Runner
    from moonshot.src.runners.runner_arguments import RunnerArguments
    from moonshot.src.runs.run import Run
    from moonshot.src.storage.storage import Storage

    results_module = load_asset("results-modules/benchmarking-result.py")
    state = SimpleNamespace(rows=[{"input": "question", "target": "old"}],
                            reports=[], metric_targets=[], progress=[])
    recipe = SimpleNamespace(id="recipe", datasets=["dataset"],
                             prompt_templates=[], metrics=["capture"], grading_scale={})

    class CaptureMetric:
        async def get_results(self, prompts, predictions, targets):
            state.metric_targets.append(copy.deepcopy(targets))
            return {"grading_criteria": {}}

    monkeypatch.setattr(cache.module.Recipe, "load", lambda name: recipe)
    monkeypatch.setattr(cache.module.Dataset, "read", lambda name: SimpleNamespace(
        num_of_dataset_prompts=len(state.rows), examples=copy.deepcopy(state.rows)
    ))
    monkeypatch.setattr(cache.module.Metric, "load", lambda name: CaptureMetric())
    monkeypatch.setattr(cache.module.ConnectorEndpoint, "read", lambda name: name)
    monkeypatch.setattr(cache.module.Connector, "create", lambda args: cache.connector)
    monkeypatch.setattr(Run, "_load_module", lambda self, key, env: (
        cache.module.Benchmarking() if key == "runner_processing_module"
        else results_module.BenchmarkingResult()
    ))
    monkeypatch.setattr(Storage, "get_filepath", lambda *args: str(tmp_path / "results.json"))
    monkeypatch.setattr(Storage, "create_object", lambda env, name, value, ext: (
        state.reports.append(copy.deepcopy(value))
    ))
    state.runner = Runner(RunnerArguments(
        id="offline-runner", name="offline runner", endpoints=[cache.connector.id],
        database_file=cache.db.db_path, database_instance=cache.db,
        progress_callback_func=state.progress.append,
    ))

    def run(system="", seed=0):
        before = len(state.reports)
        asyncio.run(state.runner.run_recipes(["recipe"], random_seed=seed, system_prompt=system))
        assert state.progress[-1]["current_error_messages"] == []
        assert len(state.reports) == before + 1
        return state.reports[-1]

    state.run = run
    return state


def report_data(report):
    return report["results"]["recipes"][0]["details"][0]["data"]


def test_public_runner_system_prompt_change_and_identical_rerun(cache, public_runner):
    first = public_runner.run("system A")
    second = public_runner.run("system B")
    third = public_runner.run("system B")
    assert len(cache.connector.calls) == 2
    assert report_data(first)[0]["predicted_result"]["response"] == "system A|question"
    assert report_data(second)[0]["predicted_result"]["response"] == "system B|question"
    assert report_data(third) == report_data(second)
    assert second["metadata"]["system_prompt"] == "system B"
    assert cache.db.sqlite_conn.execute("SELECT COUNT(*) FROM run_table").fetchone()[0] == 3


def test_public_runner_corrected_target_reaches_metric_and_report(cache, public_runner):
    public_runner.run(seed=1)
    public_runner.rows = [{"input": "question", "target": "True"}]
    report = public_runner.run(seed=29)
    assert len(cache.connector.calls) == 1
    assert public_runner.metric_targets == [["old"], ["True"]]
    assert report_data(report)[0]["target"] == "True"
    assert report["metadata"]["random_seed"] == 29


def test_public_runner_duplicate_prompts_keep_each_rows_target(cache, public_runner):
    # Prime the real persistent cache, then consume two distinct dataset rows.
    public_runner.run()
    public_runner.rows = [{"input": "question", "target": "A"},
                          {"input": "question", "target": "B"}]
    report = public_runner.run()
    assert len(cache.connector.calls) == 1
    assert public_runner.metric_targets[-1] == ["A", "B"]
    assert [row["target"] for row in report_data(report)] == ["A", "B"]
    assert report["results"]["recipes"][0]["total_num_of_prompts"] == 2
