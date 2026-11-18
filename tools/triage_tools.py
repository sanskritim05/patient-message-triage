"""Triage tools. Nurse approval is enforced separately by the hook."""

import json

from strands import ToolContext, tool


URGENT_KEYWORDS = [
    "chest pain", "shortness of breath", "can't breathe", "cannot breathe",
    "difficulty breathing", "fainted", "passed out", "slurred speech",
    "face drooping", "severe bleeding", "suicidal", "worst headache",
]

ADMIN_KEYWORDS = [
    "reschedule", "move my appointment", "cancel my appointment",
    "change my appointment", "billing", "invoice", "parking",
    "office hours", "update my address",
]

# Mock clinic schedule used by reschedule_appointment
OPEN_SLOTS = {
    "monday": "Monday 10:30 AM",
    "tuesday": "Tuesday 2:00 PM",
    "wednesday": "Wednesday 9:00 AM",
    "thursday": "Thursday 3:30 PM",
    "friday": "Friday 11:00 AM",
}


@tool(context=True)
def classify_message(message: str, tool_context: ToolContext) -> str:
    """
    Classify a patient portal message as 'urgent', 'admin', or 'clinical'.

    Always call this tool first.

    Args:
        message: The patient's message, copied exactly as written.

    Returns:
        JSON with the category, matched urgent keywords, and the next step.
    """
    # Classify the patient's original message when the app has stored it.
    # Small models can truncate the copy they pass in (for example at an
    # apostrophe), which could hide words like "chest pain".
    original = tool_context.agent.state.get("patient_message")
    text = (original or message).lower()

    urgent_hits = [kw for kw in URGENT_KEYWORDS if kw in text]
    if urgent_hits:
        category = "urgent"
    elif any(kw in text for kw in ADMIN_KEYWORDS):
        category = "admin"
    else:
        category = "clinical"

    # The approval hook uses this category, not the model's judgment.
    tool_context.agent.state.set("category", category)

    # Explicit next steps help small models reliably call the next tool.
    next_steps = {
        "admin": "Call reschedule_appointment(patient_name=..., requested_day=...).",
        "clinical": (
            "Call send_patient_reply(patient_name=..., reply=...) with a short, "
            "general reply. Do not give medical advice."
        ),
        "urgent": "Call escalate_to_care_team(patient_name=..., reported_symptoms=...) NOW.",
    }

    return json.dumps({
        "category": category,
        "urgent_keywords": urgent_hits,
        "next_step": next_steps[category],
    })


@tool
def reschedule_appointment(patient_name: str, requested_day: str) -> str:
    """
    Move a patient's appointment to the first open slot on the requested day.

    Args:
        patient_name: The patient's full name, for example "Bruce Banner".
        requested_day: The weekday the patient asked for, for example "Friday".

    Returns:
        JSON with status 'rescheduled' and the new slot, or 'unavailable'.
    """
    slot = OPEN_SLOTS.get(requested_day.strip().lower())
    if not slot:
        return json.dumps({
            "status": "unavailable",
            "requested_day": requested_day,
            "open_days": list(OPEN_SLOTS.values()),
            "next_step": (
                "Call send_patient_reply(patient_name=..., reply=...) listing the open days."
            ),
        })

    return json.dumps({
        "status": "rescheduled",
        "patient_name": patient_name,
        "new_slot": slot,
        "next_step": (
            "Call send_patient_reply(patient_name=..., reply=...) "
            f"confirming the new appointment: {slot}."
        ),
    })


@tool(context=True)
def escalate_to_care_team(
    patient_name: str, reported_symptoms: str, tool_context: ToolContext
) -> str:
    """
    Page the on-call care team about a patient reporting urgent symptoms.

    Use this for every message classified as 'urgent', before drafting any reply.

    Args:
        patient_name: The patient's full name.
        reported_symptoms: The symptoms the patient described, in their words.

    Returns:
        JSON confirming the alert was sent, and the next step.
    """
    print(f"\n🚨 [care team paged] {patient_name}: {reported_symptoms}")

    # The approval hook blocks urgent replies until this flag is set.
    tool_context.agent.state.set("care_team_paged", True)

    return json.dumps({
        "status": "escalated",
        "patient_name": patient_name,
        "alert": f"On-call nurse paged about: {reported_symptoms}",
        "next_step": (
            "Call send_patient_reply(patient_name=..., reply=...) telling the "
            "patient to call 911 or go to the nearest emergency room now, and "
            "that the care team was alerted."
        ),
    })


@tool(context=True)
def send_patient_reply(patient_name: str, reply: str, tool_context: ToolContext) -> str:
    """
    Send a reply to the patient through the portal.

    Depending on the message category, a nurse may review, edit, or reject
    the reply before it is sent.

    Args:
        patient_name: The patient's full name.
        reply: The full text of the reply to the patient.

    Returns:
        JSON confirming the exact text that was sent.
    """
    print(f"\n📨 [sent to {patient_name}] {reply}")

    # The runner checks this flag rather than trusting the model's "DONE".
    tool_context.agent.state.set("reply_sent", True)

    return json.dumps({"status": "sent", "patient_name": patient_name, "reply": reply})
