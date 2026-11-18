"""Run patient message scenarios with interactive or scripted nurse review.

Usage:
  python main.py            # all scenarios, you play the nurse
  python main.py 1          # one scenario, you play the nurse
  python main.py 1 --auto   # one scenario, scripted nurse decisions
"""

import argparse
import json

from agents import build_triage_agent
from config import get_model


SAMPLE_MESSAGES = [
    # Scenario A: admin request, handled with no human involved
    {
        "label": "Scenario A — Reschedule request (no review needed)",
        "patient": "Bruce Banner",
        "message": (
            "Hi, this is Bruce Banner. I need to reschedule my appointment "
            "to Friday if possible. Thanks!"
        ),
        "scripted_nurse": [],
    },
    # Scenario B: clinical question, nurse rejects the first draft, then
    # finishes the agent's revision by hand
    {
        "label": "Scenario B — Medication question (nurse rejects, then edits)",
        "patient": "Natasha Romanoff",
        "message": (
            "Hi, this is Natasha Romanoff. I have a bad headache. Is it okay "
            "to take ibuprofen with my lisinopril?"
        ),
        "scripted_nurse": [
            {
                "action": "reject",
                "feedback": (
                    "Do not say whether the combination is safe. Ask her to "
                    "avoid ibuprofen until a nurse calls her back today, and "
                    "mention acetaminophen is often an alternative to ask about."
                ),
            },
            {
                "action": "edit",
                "reply": (
                    "Hi Natasha, thank you for reaching out. Please hold off on "
                    "ibuprofen for now. A nurse will call you back today to go "
                    "over your medications. Acetaminophen is often an option for "
                    "headaches, and you can ask the nurse about it when they call."
                ),
            },
        ],
    },
    # Scenario C: urgent symptoms, auto-escalation, nurse edits the safety reply
    {
        "label": "Scenario C — Chest pain (escalated, nurse edits reply)",
        "patient": "Clint Barton",
        "message": (
            "This is Clint Barton. I've had chest pain and shortness of breath "
            "since this morning. Should I come in tomorrow?"
        ),
        "scripted_nurse": [
            {
                "action": "edit",
                "reply": (
                    "Clint, please call 911 or go to the nearest emergency room "
                    "now. Do not wait for tomorrow or drive yourself. Our care "
                    "team has been alerted, but do not wait for a callback "
                    "before seeking help."
                ),
            },
        ],
    },
]


def show_review_request(reason: dict) -> None:
    print("\n" + "─" * 70)
    print(f"⏸  NURSE REVIEW REQUIRED  [{reason['priority']}]")
    print("─" * 70)
    print(f"Patient : {reason['patient']}")
    print(f"Category: {reason['category']}")
    print(f"Draft   : {reason['draft_reply']}")
    print("─" * 70)


def ask_nurse() -> dict:
    """Collect a decision from a human at the terminal."""
    while True:
        choice = input("[a]pprove, [e]dit, or [r]eject? ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("e", "edit"):
            return {"action": "edit", "reply": input("Your reply: ").strip()}
        if choice in ("r", "reject"):
            return {"action": "reject", "feedback": input("Feedback for the agent: ").strip()}


def scripted_nurse(decisions: list[dict]):
    """Return a function that replays pre-written nurse decisions in order."""
    queue = iter(decisions)

    def decide() -> dict:
        decision = next(queue, {"action": "approve"})
        print(f"[scripted nurse] {decision}")
        return decision

    return decide


MAX_NUDGES = 2

NUDGE = (
    "You have not called send_patient_reply yet, so the patient has received "
    "nothing. Write a fresh reply with no apostrophes or contractions, and call "
    "send_patient_reply(patient_name=..., reply=...) now."
)


def last_next_step(messages: list) -> str | None:
    """Return the next_step from the most recent tool result, if it has one."""
    for message in reversed(messages):
        for block in reversed(message.get("content", [])):
            result = block.get("toolResult")
            if result is None:
                continue
            for item in result.get("content", []):
                try:
                    step = json.loads(item.get("text", "")).get("next_step")
                except (ValueError, AttributeError):
                    step = None
                if step:
                    return step
            return None  # The latest tool result had no next_step.
    return None


def build_nudge(agent) -> str:
    """Point the agent at the step it skipped, not straight at the send tool."""
    step = last_next_step(agent.messages)
    if not step:
        return NUDGE
    return (
        "You stopped before finishing, so the patient has received nothing. "
        f"Do this next: {step} Keep going until send_patient_reply returns "
        '"status": "sent". Write replies with no apostrophes or contractions.'
    )


def resolve_interrupts(agent, result, decide):
    """While the agent is paused, collect nurse decisions and resume it."""
    while result.stop_reason == "interrupt":
        responses = []
        for interrupt in result.interrupts:
            show_review_request(interrupt.reason)
            responses.append({
                "interruptResponse": {
                    "interruptId": interrupt.id,
                    "response": decide(),
                }
            })
        # Resume the SAME agent with the nurse's answers.
        result = agent(responses)
    return result


def run_message(model, scenario: dict, auto: bool) -> None:
    print("\n" + "=" * 70)
    print(f"  {scenario['label']}")
    print("=" * 70)
    print(f"Message: {scenario['message']}")

    decide = scripted_nurse(scenario["scripted_nurse"]) if auto else ask_nurse
    agent = build_triage_agent(model)
    # The classifier reads this instead of trusting the model's copy.
    agent.state.set("patient_message", scenario["message"])

    # A model's "DONE" does not prove it called the send tool.
    for attempt in range(MAX_NUDGES + 1):
        if attempt:
            print(f"\n↻ [guard] no reply was sent, nudging the agent ({attempt}/{MAX_NUDGES})")
        prompt = build_nudge(agent) if attempt else scenario["message"]
        result = resolve_interrupts(agent, agent(prompt), decide)
        if agent.state.get("reply_sent"):
            print(f"\n✅ {str(result).strip()}")
            return

    print("\n⚠️  No reply was sent. Manual nurse follow-up is required.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scenario",
        type=int,
        nargs="?",
        choices=range(len(SAMPLE_MESSAGES)),
        help="scenario number (default: run all)",
    )
    parser.add_argument("--auto", action="store_true", help="use scripted nurse decisions")
    args = parser.parse_args()
    scenarios = (
        SAMPLE_MESSAGES if args.scenario is None else [SAMPLE_MESSAGES[args.scenario]]
    )

    model = get_model()
    for scenario in scenarios:
        run_message(model, scenario, args.auto)


if __name__ == "__main__":
    main()
