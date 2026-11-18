"""Exercise nurse decisions and audit logging without an agent or model."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hooks import NurseApprovalHook
from hooks import nurse_approval


@pytest.mark.parametrize("action", ["approve", "edit", "reject"])
def test_review_applies_decision_and_audits_original_draft(tmp_path, monkeypatch, action):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(nurse_approval, "AUDIT_LOG_PATH", audit_path)
    draft = "Thank you for your message. A nurse will follow up."
    decision = {"action": action, "reply": "Nurse edited reply", "feedback": "Revise"}
    event = SimpleNamespace(
        agent=SimpleNamespace(state={"category": "urgent", "care_team_paged": True}),
        tool_use={
            "name": "send_patient_reply",
            "input": {"patient_name": "Demo Patient", "reply": draft},
        },
        interrupt=Mock(return_value=decision),
        cancel_tool=False,
    )

    NurseApprovalHook().review_reply(event)

    assert event.interrupt.call_args.kwargs["reason"]["priority"] == "URGENT"
    if action == "reject":
        assert "REJECTED BY NURSE" in event.cancel_tool
        assert "Revise" in event.cancel_tool
    else:
        assert event.cancel_tool is False
    expected_reply = decision["reply"] if action == "edit" else draft
    assert event.tool_use["input"]["reply"] == expected_reply
    entry = json.loads(audit_path.read_text(encoding="utf-8"))
    assert entry["agent_draft"] == draft
    assert entry["nurse_action"] == action
    assert entry["patient"] == "Demo Patient"


def test_urgent_reply_is_blocked_until_care_team_is_paged(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(nurse_approval, "AUDIT_LOG_PATH", audit_path)
    event = SimpleNamespace(
        agent=SimpleNamespace(state={"category": "urgent"}),
        tool_use={
            "name": "send_patient_reply",
            "input": {"reply": "Please call 911 or go to the nearest emergency room now."},
        },
        interrupt=Mock(),
        cancel_tool=False,
    )

    NurseApprovalHook().review_reply(event)

    assert "escalate_to_care_team" in event.cancel_tool
    event.interrupt.assert_not_called()
    assert not audit_path.exists()


def test_truncated_reply_is_cancelled_before_nurse_review(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(nurse_approval, "AUDIT_LOG_PATH", audit_path)
    event = SimpleNamespace(
        agent=SimpleNamespace(state={"category": "clinical"}),
        tool_use={"name": "send_patient_reply", "input": {"reply": "Hi"}},
        interrupt=Mock(),
        cancel_tool=False,
    )

    NurseApprovalHook().review_reply(event)

    assert "AUTO-CHECK FAILED" in event.cancel_tool
    event.interrupt.assert_not_called()
    assert not audit_path.exists()
