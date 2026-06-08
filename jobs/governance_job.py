"""Nebius Job #1: Scheduled Governance Evaluation.

Autonomously evaluates current production behavior and generates a compliance
report. Intended for a nightly schedule such as 00:00 UTC.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from aurexis_twin import run_governance_cycle, write_json_artifact


def main() -> None:
    result = run_governance_cycle(model_name="scheduled_rf", jurisdiction="EU AI Act")
    output_path = write_json_artifact("scheduled_governance_evaluation", asdict(result))
    print(f"scheduled_governance_evaluation={output_path}")
    print(asdict(result))


if __name__ == "__main__":
    main()

