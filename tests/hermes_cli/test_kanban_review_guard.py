"""Kanban P0 review guard (card t_159b0030, incident run 9018).

Run 9018: the review lane spawned the card's own builder (``kublai``) as
its reviewer; the builder approved, merged, and deployed its own work
before the tester (``orda``) reviewed it.

Covers:

a. The review lane never assigns a builder (assignee at handoff,
   ``created_by``, or any profile that ran an implementation worker).
   With no eligible reviewer the card stays in review unassigned with an
   event and a comment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def cfg(monkeypatch):
    """Mutable stand-in for the ``kanban`` config section."""
    section: dict = {}
    monkeypatch.setattr(kb, "_kanban_config_section", lambda: section)
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: True)
    return section


@pytest.fixture
def spawns(all_assignees_spawnable):
    calls: list = []

    def fake_spawn(task, workspace, board=None):
        calls.append((task.id, task.assignee))
        return 4242

    fake_spawn.calls = calls
    return fake_spawn


def _events(conn, tid, kind):
    return [
        json.loads(r["payload"]) if r["payload"] else {}
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        )
    ]


def _assignee(conn, tid):
    return conn.execute(
        "SELECT assignee FROM tasks WHERE id = ?", (tid,),
    ).fetchone()["assignee"]


def _built_and_handed_to_review(conn, *, builder="kublai", created_by="ogedei",
                                reviewer=None):
    tid = kb.create_task(conn, title="p0 card", assignee=builder,
                         created_by=created_by)
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert kb.request_review(
        conn, tid, summary="built", expected_run_id=claimed.current_run_id,
        reviewer=reviewer,
    )
    return tid


def _orda_review_run(conn, cfg, spawns):
    """Builder hands off, dispatcher spawns orda for the review run."""
    tid = _built_and_handed_to_review(conn)
    res = kb.dispatch_once(conn, spawn_fn=spawns)
    assert [s[:2] for s in res.spawned] == [(tid, "orda")]
    return tid


# ---------------------------------------------------------------------------
# a. reviewer selection
# ---------------------------------------------------------------------------


def test_incident_9018_builder_is_replaced_by_orda(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn)
        # Pre-fix the card sat in review still assigned to its builder.
        assert _assignee(conn, tid) == "kublai"
        res = kb.dispatch_once(conn, spawn_fn=spawns)
        assert spawns.calls == [(tid, "orda")]
        assert res.review_reassigned == [(tid, "kublai", "orda")]
        assert _assignee(conn, tid) == "orda"
        ev = _events(conn, tid, kb.REVIEW_REASSIGNED_EVENT)
        assert ev and ev[0]["from"] == "kublai" and ev[0]["to"] == "orda"
        assert "kublai" in ev[0]["builders"]
        run = kb.latest_run(conn, tid)
        assert run.profile == "orda"


def test_no_eligible_reviewer_leaves_card_unassigned_in_review(
    kanban_home, cfg, spawns,
):
    cfg["reviewer_profiles"] = ["kublai"]  # only the builder is configured
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn)
        res = kb.dispatch_once(conn, spawn_fn=spawns)
        assert spawns.calls == []
        assert res.review_no_eligible_reviewer == [tid]
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee is None
        assert len(_events(conn, tid, kb.REVIEWER_UNAVAILABLE_EVENT)) == 1
        comments = kb.list_comments(conn, tid)
        assert [c.author for c in comments] == [kb.REVIEW_GUARD_AUTHOR]
        assert "cannot review" in comments[0].body

        # Next tick: stays parked, no duplicate comment, still no spawn.
        res2 = kb.dispatch_once(conn, spawn_fn=spawns)
        assert spawns.calls == []
        assert tid in res2.skipped_unassigned
        assert len(kb.list_comments(conn, tid)) == 1


def test_created_by_and_run_history_profiles_are_excluded(
    kanban_home, cfg, spawns,
):
    cfg["reviewer_profiles"] = ["orda", "jochi", "chagatai"]
    with kb.connect() as conn:
        # Filed by orda; first implementation attempt by jochi, then kublai.
        tid = kb.create_task(conn, title="x", assignee="jochi", created_by="orda")
        first = kb.claim_task(conn, tid)
        assert first is not None
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "current_run_id=NULL WHERE id=?", (tid,),
        )
        conn.execute(
            "UPDATE task_runs SET status='reclaimed', ended_at=started_at "
            "WHERE id=?", (first.current_run_id,),
        )
        assert kb.assign_task(conn, tid, "kublai")
        second = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", expected_run_id=second.current_run_id,
        )
        assert kb.card_builder_profiles(conn, tid) == {"orda", "jochi", "kublai"}
        kb.dispatch_once(conn, spawn_fn=spawns)
        assert spawns.calls == [(tid, "chagatai")]


def test_reviewer_can_rereview_after_requesting_changes(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _orda_review_run(conn, cfg, spawns)
        ok, implementer = kb.request_changes(conn, tid, reason="fix it")
        assert ok and implementer == "kublai"
        rework = kb.claim_task(conn, tid)
        assert rework.assignee == "kublai"
        assert kb.request_review(
            conn, tid, summary="fixed", expected_run_id=rework.current_run_id,
        )
        assert _assignee(conn, tid) == "orda"  # persisted reviewer provenance
        assert "orda" not in kb.card_builder_profiles(conn, tid)
        assert kb.claim_review_task(conn, tid) is not None


def test_claim_review_task_refuses_explicit_self_review(kanban_home, cfg):
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn, reviewer="kublai")
        assert kb.claim_review_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "review"
        refused = _events(conn, tid, kb.REVIEW_CLAIM_REFUSED_EVENT)
        assert refused and refused[0]["assignee"] == "kublai"


def test_explicit_independent_reviewer_is_kept(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn, reviewer="chagatai")
        res = kb.dispatch_once(conn, spawn_fn=spawns)
        assert spawns.calls == [(tid, "chagatai")]
        assert res.review_reassigned == []


def test_dry_run_reports_but_does_not_mutate(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn)
        res = kb.dispatch_once(conn, spawn_fn=spawns, dry_run=True)
        assert res.review_reassigned == [(tid, "kublai", "orda")]
        assert res.spawned == [(tid, "orda", "")]
        assert spawns.calls == []
        assert _assignee(conn, tid) == "kublai"
        assert _events(conn, tid, kb.REVIEW_REASSIGNED_EVENT) == []
