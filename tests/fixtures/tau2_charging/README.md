`live_run4.json` contains task/configuration extracts from:

- `results/tau2/gemma4-12b-test/tasks/0002-native.json`: the telecom MMS task
  with `PERSONA:Hard`, failing on its Luna user simulator's tool-bearing call.
- `results/tau2/luna-probe/tasks/0000-native.json`: retail task `5`, failing
  on the first Luna teacher-agent call after one settled user-simulator call.

Each extract includes the last event from the corresponding `budget.json`.
Both saved simulations have empty messages and contain only the final retry's
`BudgetStopped: unknown_charge`; they do not contain the original response or
the original exception. Tests rebuild a tool-bearing request from the recorded
actor configuration, task scenario/policy, and one rubric action. Tool schemas
and all returned responses/usage shapes in these tests are synthetic.

An offline call through the installed LiteLLM with the old registration and
`tool_choice="auto"` reproduces the failure before HTTP:
`UnsupportedParamsError: openai does not support parameters: ['tool_choice'],
for model=gpt-5.6-luna`. The regression preserves parameter validation and
uses mocked HTTP, without dropping tools or making a live request.

The tests also cover omitted cache details, provider prefixes, dated aliases,
missing usage, and LiteLLM's all-zero usage placeholder for omitted HTTP usage.
Dates in synthetic model names exercise alias parsing; they are not claimed
to be snapshot names returned by the failed calls.
