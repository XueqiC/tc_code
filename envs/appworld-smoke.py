"""Run one AppWorld train task with a trivial code-emitting agent."""

from __future__ import annotations

import json

from appworld import AppWorld, load_task_ids


def main() -> None:
    task_id = load_task_ids("train")[0]
    print(f"task_id={task_id}")

    with AppWorld(task_id=task_id, experiment_name="smoke/trivial") as world:
        print(f"instruction={world.task.instruction}")

        # A trivial agent action in the documented code-string interface. It is
        # expected to score poorly, but it exercises execution and evaluation.
        code = "apis.supervisor.complete_task()"
        print(f"agent_code={code}")
        execution_output = world.execute(code)
        print(f"execution_output={execution_output}")

        evaluation = world.evaluate().to_dict()
        print("evaluation_result=" + json.dumps(evaluation, sort_keys=True))


if __name__ == "__main__":
    main()
