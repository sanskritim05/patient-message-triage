"""Pause non-admin replies for nurse approval, edits, or rejection; audit decisions.

Before a nurse sees anything, cheap automated checks run first: urgent replies
are blocked until the care team has been paged, and truncated drafts are
returned to the agent.
"""

import json
from datetime import datetime, timezone
from typing import Any

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

AUDIT_LOG_PATH = "audit_log.jsonl"
MIN_REPLY_WORDS = 8


def requires_nurse_review(tool_name: str, category: str | None) -> bool:
    """Require review for non-admin replies, including unclassified messages."""
    return tool_name == "send_patient_reply" and category != "admin"


class NurseApprovalHook(HookProvider):
    """Pause the agent for nurse review before risky patient replies."""

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.review_reply)

    def review_reply(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        category = event.agent.state.get("category")

        if not requires_nurse_review(tool_name, category):
            return

        tool_input = event.tool_use["input"]

        # An urgent reply must never go out before the care team is paged.
        if category == "urgent" and not event.agent.state.get("care_team_paged"):
            event.cancel_tool = (
                "NOT SENT: this message is urgent and the care team has not been "
                "paged yet. Call escalate_to_care_team(patient_name=..., "
                "reported_symptoms=...) first, then call "
                "send_patient_reply(patient_name=..., reply=...) again."
            )
            return

        # Return likely truncated drafts to the agent before asking a nurse.
        draft = tool_input.get("reply") or ""
        if len(draft.split()) < MIN_REPLY_WORDS:
            event.cancel_tool = (
                "AUTO-CHECK FAILED: the reply was cut off and was NOT sent. "
                "Write the complete reply with no apostrophes or contractions "
                "and call send_patient_reply(patient_name=..., reply=...) again."
            )
            return

        # Interrupt on first call; return the nurse's answer when resumed.
        decision = event.interrupt(
            "nurse_review",
            reason={
                "patient": tool_input.get("patient_name"),
                "category": category,
                "priority": "URGENT" if category == "urgent" else "routine",
                "draft_reply": draft,
            },
        )

        action = decision.get("action")

        if action == "edit":
            event.tool_use = {
                **event.tool_use,
                "input": {**tool_input, "reply": decision["reply"]},
            }
        elif action == "reject":
            event.cancel_tool = (
                "REJECTED BY NURSE. The reply was NOT sent. "
                f"Nurse feedback: {decision.get('feedback', '')} "
                "Write a new reply that follows this feedback and call "
                "send_patient_reply(patient_name=..., reply=...) again."
            )

        _write_audit_entry(
            patient=tool_input.get("patient_name"),
            category=category,
            draft=draft,
            decision=decision,
        )


def _write_audit_entry(
    patient: str | None, category: str | None, draft: str, decision: dict
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient": patient,
        "category": category,
        "agent_draft": draft,
        "nurse_action": decision.get("action"),
        "nurse_reply": decision.get("reply"),
        "nurse_feedback": decision.get("feedback"),
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
