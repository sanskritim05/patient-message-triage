"""
Tests for the approval policy. No LLM, no Ollama, runs in milliseconds.
This is the payoff of keeping "when to pause" in Python instead of a prompt.
"""

import json
from types import SimpleNamespace

from hooks import requires_nurse_review
from tools.triage_tools import classify_message


def test_admin_replies_go_out_automatically():
    assert requires_nurse_review("send_patient_reply", "admin") is False


def test_clinical_and_urgent_replies_need_a_nurse():
    assert requires_nurse_review("send_patient_reply", "clinical") is True
    assert requires_nurse_review("send_patient_reply", "urgent") is True


def test_unclassified_reply_fails_safe():
    assert requires_nurse_review("send_patient_reply", None) is True


def test_non_reply_tools_never_pause():
    for tool in ("classify_message", "reschedule_appointment", "escalate_to_care_team"):
        assert requires_nurse_review(tool, "urgent") is False


class FakeState(dict):
    def set(self, key, value):
        self[key] = value


def _classify(message: str) -> str:
    ctx = SimpleNamespace(agent=SimpleNamespace(state=FakeState()))
    return json.loads(classify_message._tool_func(message, tool_context=ctx))["category"]


def test_classifier_categories():
    assert _classify("Can I reschedule to Friday?") == "admin"
    assert _classify("Is ibuprofen okay with lisinopril?") == "clinical"
    assert _classify("I have chest pain and shortness of breath") == "urgent"


def test_urgent_beats_admin():
    assert _classify("I want to reschedule but I have chest pain") == "urgent"


def test_classifier_uses_original_message_not_model_copy():
    # A real 3B-model run passed "Clint Barton. I" (cut off at "I've"),
    # which hid "chest pain" and would have skipped the care team page.
    state = FakeState(patient_message="I've had chest pain since this morning.")
    ctx = SimpleNamespace(agent=SimpleNamespace(state=state))
    result = json.loads(classify_message._tool_func("Clint Barton. I", tool_context=ctx))
    assert result["category"] == "urgent"
