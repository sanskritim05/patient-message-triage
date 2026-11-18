"""Check retry limits and nurse review without calling an LLM."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import main


class FakeState(dict):
    def set(self, key, value):
        self[key] = value


def test_runner_stores_the_original_message_for_the_classifier(monkeypatch):
    agent = Mock(return_value=SimpleNamespace(stop_reason="end_turn"))
    agent.state = FakeState(reply_sent=True)
    agent.messages = []
    monkeypatch.setattr(main, "build_triage_agent", lambda model: agent)

    main.run_message(None, {"label": "t", "message": "Hello", "scripted_nurse": []}, auto=True)

    assert agent.state["patient_message"] == "Hello"


@pytest.mark.parametrize("send_on_call", [1, 2, 3, None])
def test_runner_retries_until_sent_or_limit(monkeypatch, capsys, send_on_call):
    state = FakeState()
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        if len(calls) == send_on_call:
            state["reply_sent"] = True
        return SimpleNamespace(stop_reason="end_turn")

    agent = Mock(side_effect=invoke)
    agent.state = state
    agent.messages = []
    monkeypatch.setattr(main, "build_triage_agent", lambda model: agent)
    scenario = {"label": "test", "message": "Hello", "scripted_nurse": []}

    main.run_message(None, scenario, auto=True)

    expected_calls = send_on_call or main.MAX_NUDGES + 1
    assert calls == ["Hello"] + [main.NUDGE] * (expected_calls - 1)
    assert ("Manual nurse follow-up is required" in capsys.readouterr().out) == (
        send_on_call is None
    )


def _tool_result(text: str) -> dict:
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "status": "success",
                                    "content": [{"text": text}]}}],
    }


def test_nudge_repeats_the_skipped_next_step():
    # The model classified an admin message, then claimed "DONE" without
    # rescheduling. The nudge must send it back to reschedule_appointment.
    step = "Call reschedule_appointment(patient_name=..., requested_day=...)."
    agent = SimpleNamespace(messages=[
        _tool_result(json.dumps({"category": "admin", "next_step": step})),
        {"role": "assistant", "content": [{"text": "DONE: Rescheduled for Friday."}]},
    ])

    nudge = main.build_nudge(agent)

    assert step in nudge
    assert nudge != main.NUDGE


def test_nudge_falls_back_without_a_next_step():
    rejected = _tool_result("REJECTED BY NURSE. The reply was NOT sent.")
    assert main.build_nudge(SimpleNamespace(messages=[rejected])) == main.NUDGE
    assert main.build_nudge(SimpleNamespace(messages=[])) == main.NUDGE


def test_resolve_interrupts_resumes_each_round_with_nurse_decisions(monkeypatch):
    monkeypatch.setattr(main, "show_review_request", Mock())
    first = SimpleNamespace(
        stop_reason="interrupt",
        interrupts=[SimpleNamespace(id="first", reason={})],
    )
    second = SimpleNamespace(
        stop_reason="interrupt",
        interrupts=[SimpleNamespace(id="second", reason={})],
    )
    done = SimpleNamespace(stop_reason="end_turn")
    agent = Mock(side_effect=[second, done])
    reject = {"action": "reject", "feedback": "Please revise"}
    edit = {"action": "edit", "reply": "Reviewed reply"}
    decide = main.scripted_nurse([reject, edit])

    assert main.resolve_interrupts(agent, first, decide) is done
    assert agent.call_args_list == [
        call([{"interruptResponse": {"interruptId": "first", "response": reject}}]),
        call([{"interruptResponse": {"interruptId": "second", "response": edit}}]),
    ]
    assert decide() == {"action": "approve"}


@pytest.mark.parametrize("argument", ["-1", "3", "invalid", "--unknown"])
def test_invalid_cli_arguments_exit_before_loading_model(monkeypatch, argument):
    monkeypatch.setattr("sys.argv", ["main.py", argument])
    get_model = Mock()
    monkeypatch.setattr(main, "get_model", get_model)

    with pytest.raises(SystemExit) as error:
        main.main()

    assert error.value.code == 2
    get_model.assert_not_called()
