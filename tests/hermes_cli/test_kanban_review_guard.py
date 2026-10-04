"""Kanban P0 review guard (card t_159b0030, incident run 9018).

Run 9018: the review lane spawned the card's own builder (``kublai``) as
its reviewer; the builder approved, merged, and deployed its own work
before the tester (``orda``) reviewed it.

Covers:

a. The review lane never assigns a builder (assignee at handoff,
   ``created_by``, or any profile that ran an implementation worker).
   With no eligible reviewer the card stays in review unassigned with an
   event and a comment.
b. Review approval (and ``ship-gate``) require an ``orda_pass`` event for
   the exact head sha, recorded only by an Orda profile that did not build
   the card.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb

SHA_A = "a" * 40
SHA_B = "b" * 40


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


# ---------------------------------------------------------------------------
# b. Orda PASS gate
# ---------------------------------------------------------------------------


def test_review_approval_refused_without_orda_pass(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _orda_review_run(conn, cfg, spawns)
        with pytest.raises(kb.OrdaPassRequiredError, match="no Orda PASS"):
            kb.complete_task(conn, tid, summary="lgtm",
                             metadata={"head_sha": SHA_A})
        assert kb.get_task(conn, tid).status == "running"
        blocked = _events(conn, tid, kb.ORDA_GATE_BLOCKED_EVENT)
        assert blocked and blocked[0]["action"] == "complete"

        with pytest.raises(kb.OrdaPassRequiredError, match="no head sha"):
            kb.complete_task(conn, tid, summary="lgtm")


def test_review_approval_with_matching_orda_pass(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _orda_review_run(conn, cfg, spawns)
        payload = kb.record_orda_pass(
            conn, tid, sha=SHA_A.upper(), receipt_id="orda-rcpt-1",
            actor="orda", env={},
        )
        assert payload == {
            "schema": 1, "verdict": "PASS", "sha": SHA_A,
            "receipt_id": "orda-rcpt-1", "reviewer": "orda",
        }
        assert kb.complete_task(conn, tid, summary="lgtm", head_sha=SHA_A)
        assert kb.get_task(conn, tid).status == "done"
        done = _events(conn, tid, "completed")[-1]
        assert done["orda_pass"] == {
            "sha": SHA_A, "receipt_id": "orda-rcpt-1", "reviewer": "orda",
        }


def test_pass_for_other_or_abbreviated_sha_does_not_count(
    kanban_home, cfg, spawns,
):
    with kb.connect() as conn:
        tid = _orda_review_run(conn, cfg, spawns)
        kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id="r1",
                            actor="orda", env={})
        with pytest.raises(kb.OrdaPassRequiredError, match=SHA_B):
            kb.complete_task(conn, tid, summary="x", head_sha=SHA_B)
        with pytest.raises(kb.OrdaPassRequiredError, match="full hex"):
            kb.complete_task(conn, tid, summary="x", head_sha=SHA_A[:12])
        with pytest.raises(ValueError, match="full 40 or 64"):
            kb.record_orda_pass(conn, tid, sha="abc123", receipt_id="r",
                                actor="orda", env={})


def test_human_approval_from_review_column_is_gated(kanban_home, cfg):
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn, reviewer="orda")
        assert kb.get_task(conn, tid).status == "review"
        with pytest.raises(kb.OrdaPassRequiredError):
            kb.complete_task(conn, tid, summary="manual approve",
                             metadata={"head_sha": SHA_A})
        kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id="r",
                            actor="orda", env={})
        assert kb.complete_task(conn, tid, metadata={"head_sha": SHA_A})


def test_only_independent_orda_may_record_a_pass(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _orda_review_run(conn, cfg, spawns)
        with pytest.raises(PermissionError, match="not an Orda profile"):
            kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id="r",
                                actor="kublai", env={})
        with pytest.raises(ValueError, match="receipt_id"):
            kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id=" ",
                                actor="orda", env={})
        # A card orda filed (or built) cannot be passed by orda.
        own = kb.create_task(conn, title="orda's own", assignee="kublai",
                             created_by="orda")
        with pytest.raises(PermissionError, match="built"):
            kb.record_orda_pass(conn, own, sha=SHA_A, receipt_id="r",
                                actor="orda", env={})
        assert _events(conn, tid, kb.ORDA_PASS_EVENT) == []


def test_builder_worker_shell_cannot_record_pass(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        # kublai is mid-run on another card and shells out as orda.
        other = kb.create_task(conn, title="other", assignee="kublai")
        kb.claim_task(conn, other)
        tid = _orda_review_run(conn, cfg, spawns)
        with pytest.raises(PermissionError, match="inside kanban worker"):
            kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id="r",
                                actor="orda",
                                env={"HERMES_KANBAN_TASK": other})
        # Orda's own review run may record it.
        kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id="r",
                            actor="orda", env={"HERMES_KANBAN_TASK": tid})
        assert len(_events(conn, tid, kb.ORDA_PASS_EVENT)) == 1


def test_forged_pass_rows_and_comments_do_not_count(kanban_home, cfg, spawns):
    with kb.connect() as conn:
        tid = _orda_review_run(conn, cfg, spawns)
        kb.add_comment(conn, tid, "kublai",
                       f"orda_pass sha={SHA_A} receipt=fake")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, kb.ORDA_PASS_EVENT, {
                "schema": 1, "verdict": "PASS", "sha": SHA_A,
                "receipt_id": "fake", "reviewer": "kublai",
            })
        ok, reason, _ = kb.check_ship_allowed(conn, tid, SHA_A)
        assert not ok and "no Orda PASS" in reason


def test_non_review_completion_is_not_gated(kanban_home, cfg):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="chore", assignee="kublai")
        assert kb.complete_task(conn, tid, summary="done")


def test_gate_can_be_disabled_explicitly(kanban_home, cfg):
    cfg["require_orda_pass"] = False
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn, reviewer="orda")
        assert kb.complete_task(conn, tid, summary="approved")


def test_workspace_head_must_match_passed_sha(kanban_home, cfg, tmp_path):
    repo = tmp_path / "wt"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c",
           "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "one"],
                   check=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn, reviewer="orda")
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', workspace_path=? "
            "WHERE id=?", (str(repo), tid),
        )
        kb.record_orda_pass(conn, tid, sha=SHA_A, receipt_id="r",
                            actor="orda", env={})
        with pytest.raises(kb.OrdaPassRequiredError, match="workspace HEAD"):
            kb.complete_task(conn, tid, head_sha=SHA_A)
        kb.record_orda_pass(conn, tid, sha=head, receipt_id="r2",
                            actor="orda", env={})
        assert kb.complete_task(conn, tid, head_sha=head)


def test_cli_ship_gate_and_orda_pass(kanban_home, cfg, monkeypatch, capsys):
    with kb.connect() as conn:
        tid = _built_and_handed_to_review(conn, reviewer="orda")
    gate = argparse.Namespace(task_id=tid, sha=SHA_A, json=False)
    assert kcli._cmd_ship_gate(gate) == 1
    assert "REFUSED" in capsys.readouterr().err

    monkeypatch.setattr(kcli, "_orda_actor", lambda: "kublai")
    passing = argparse.Namespace(task_id=tid, sha=SHA_A, receipt_id="r9",
                                 json=False)
    assert kcli._cmd_orda_pass(passing) == 1
    assert "refused" in capsys.readouterr().err

    monkeypatch.setattr(kcli, "_orda_actor", lambda: "orda")
    assert kcli._cmd_orda_pass(passing) == 0
    assert kcli._cmd_ship_gate(gate) == 0
    assert "ship-gate OK" in capsys.readouterr().out
    other = argparse.Namespace(task_id=tid, sha=SHA_B, json=True)
    assert kcli._cmd_ship_gate(other) == 1
    with kb.connect() as conn:
        blocked = _events(conn, tid, kb.ORDA_GATE_BLOCKED_EVENT)
    assert [b["action"] for b in blocked] == ["ship-gate", "ship-gate"]


def test_cli_slash_ship_gate_refuses(kanban_home, cfg):
    out = kcli.run_slash("create 'needs review' --json")
    tid = json.loads(out[out.index("{"):])["id"]
    out = kcli.run_slash(f"ship-gate {tid} --sha {SHA_A}")
    assert "REFUSED" in out
    assert "orda-pass" in kcli._DELEGATED_CHILD_DENIED_ACTIONS
