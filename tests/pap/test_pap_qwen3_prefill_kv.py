# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.pap.model.prefill import _pap_prune_imported_prefill_kv


def test_pap_prune_imported_prefill_kv_removes_finished_requests() -> None:
    imported = {
        ("cmpl-finished", "model.layers.0.self_attn.attn", 1024, "handle-a"),
        ("cmpl-active", "model.layers.0.self_attn.attn", 1024, "handle-b"),
        ("cmpl-finished", "model.layers.1.self_attn.attn", 1024, "handle-a"),
    }

    _pap_prune_imported_prefill_kv(imported, ("cmpl-finished",))

    assert imported == {
        ("cmpl-active", "model.layers.0.self_attn.attn", 1024, "handle-b"),
    }


def test_pap_prune_imported_prefill_kv_ignores_empty_finished_set() -> None:
    imported = {
        ("cmpl-active", "model.layers.0.self_attn.attn", 1024, "handle-a"),
    }

    _pap_prune_imported_prefill_kv(imported, ())

    assert imported == {
        ("cmpl-active", "model.layers.0.self_attn.attn", 1024, "handle-a"),
    }


def test_pap_prune_imported_prefill_kv_accepts_empty_import_set() -> None:
    imported: set[tuple[str, str, int, str]] = set()

    _pap_prune_imported_prefill_kv(imported, ("cmpl-finished",))

    assert imported == set()
