# Patient Message Triage

A small [Strands Agents](https://github.com/strands-agents/sdk-python) demo that classifies fictional patient portal messages and pauses for nurse review before sending a non-admin reply. It runs locally with Ollama, with optional AWS Bedrock support.

**The agent drafts. A nurse decides. Python code enforces the review gate.**

Appointment changes, care team pages, and patient replies are simulated with console output. There is no connection to a patient portal, clinic schedule, paging service, or nurse inbox.

| Category | Demo action | Nurse review |
| --- | --- | --- |
| Admin | Look up a mock appointment slot and confirm it | No |
| Clinical | Draft a general acknowledgment | Approve, edit, or reject |
| Urgent | Simulate a care team page, then draft an emergency reply | Approve, edit, or reject |

## Setup

Requires Python 3.10 or later and [Ollama](https://ollama.com) for the default provider.

```sh
git clone https://github.com/sanskritim05/patient-message-triage.git
cd patient-message-triage
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Start Ollama in a separate terminal if it is not already running:

```sh
ollama serve
```

Then download the default model:

```sh
ollama pull llama3.2
```

To use another local model, set `OLLAMA_MODEL` in `.env`. Set `OLLAMA_HOST` if your server uses a different address.

For AWS Bedrock, set `USE_BEDROCK=true`, `BEDROCK_MODEL_ID`, and `AWS_REGION` in `.env`, and configure standard AWS credentials with access to that model. Bedrock requests use your AWS account and may incur charges.

## Run

```sh
python main.py            # all scenarios, interactive nurse decisions
python main.py 1          # scenario B, interactive nurse decisions
python main.py 1 --auto   # scenario B, scripted nurse decisions
python main.py --help
```

| Number | Scenario | Scripted review (`--auto`) |
| --- | --- | --- |
| 0 | Bruce Banner asks to reschedule to Friday | No review |
| 1 | Natasha Romanoff asks about ibuprofen with lisinopril | Reject the first draft, then edit the revision |
| 2 | Clint Barton reports chest pain and shortness of breath | Edit the urgent reply |

During interactive review, enter `a` to approve, `e` to replace the draft, or `r` to reject it with feedback. Every returned nurse decision is appended to `audit_log.jsonl` in the current directory. Auto mode replays the example decisions and defaults to approval once they are exhausted; it is only for demonstrations.

## How it works

1. `classify_message` applies keyword rules and saves the category in agent state. Urgent matches take priority over admin matches; other messages default to clinical. It classifies the original patient message, which the runner stores in agent state, rather than the copy the model passes in, because small models can truncate that copy.
2. The agent calls tools to simulate an appointment change or urgent escalation, then drafts a reply.
3. `NurseApprovalHook` intercepts `send_patient_reply`. Admin replies bypass review; clinical, urgent, and unclassified replies require it. Before review, urgent replies are returned to the agent until `escalate_to_care_team` has run, and drafts under eight words are returned as potentially truncated.
4. The runner displays each interrupt and resumes the same agent with the nurse's decision. Approval allows the draft, an edit replaces it, and rejection cancels the send with feedback for revision.
5. The runner checks the `reply_sent` state flag instead of trusting the model's final text. If needed, it nudges the agent up to twice, repeating the `next_step` from the latest tool result so a skipped step, such as rescheduling, is not bypassed. If no reply is sent, it prints that manual follow-up is required.

Tool results include explicit `next_step` instructions to help small local models follow the workflow. Draft instructions avoid apostrophes and contractions as a workaround for truncated tool arguments observed in this demo.

## Tests

Tests run without Ollama, AWS credentials, or model requests:

```sh
python -m pytest -q
```

GitHub Actions runs the suite on Python 3.10 and 3.14 for pushes and pull requests.

## Files

```text
agents/triage_agent.py       Agent configuration and system prompt
hooks/nurse_approval.py      Review policy, interrupts, and audit logging
tools/triage_tools.py        Classification and simulated clinic actions
tests/                      Policy, hook, classifier, and runner tests
config.py                   Ollama or Bedrock model selection
main.py                     Fictional scenarios and CLI runner
.env.example                Example configuration without credentials
.github/workflows/tests.yml Automated tests
```

## Demo limitations

This is an educational demo, not a medical device. Use fictional messages only. The keyword classifier does not understand negation, mixed clinical and admin requests, or symptoms outside its keyword list. The classifier reads the original message from state, and the hook blocks urgent replies until the care team page has run, but nothing forces the model to call `classify_message` first; an unclassified reply still requires review. The eight-word check cannot establish that a draft is complete or medically correct.

The sample nurse decisions are fictional and do not arrange real callbacks. Local `.env` files, audit logs, virtual environments, caches, and the working blog draft are excluded by `.gitignore`.
