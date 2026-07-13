from pathlib import Path


ROOT = Path(__file__).parents[2]
ORCHESTRATOR = (
    ROOT
    / ".claude/skills/vllm-pap-benchmark/scripts/"
    / "run_pd_pap_multiturn_load.sh"
)


def test_default_testbed_contains_three_named_lanes() -> None:
    text = ORCHESTRATOR.read_text(encoding="utf-8")

    assert "pd-oneway" in text
    assert "pd-twoway" in text
    assert "pd_oneway_aggregate.json" in text
    assert "pd_twoway_aggregate.json" in text
    assert "compare-three" in text


def test_formal_testbed_uses_three_by_three_latin_square() -> None:
    normalized = " ".join(ORCHESTRATOR.read_text(encoding="utf-8").split())

    assert "pd_oneway pd_twoway pap" in normalized
    assert "pd_twoway pap pd_oneway" in normalized
    assert "pap pd_oneway pd_twoway" in normalized
