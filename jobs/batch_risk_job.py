"""Nebius Job #3: Batch Enterprise Risk Assessment.

Evaluates an enterprise portfolio of AI systems overnight.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from aurexis_twin import batch_risk_assessment, write_json_artifact


def main() -> None:
    result = batch_risk_assessment()
    payload = {
        "timestamp": result.timestamp,
        "summary": result.summary,
        "report_path": result.report_path,
        "models": [asdict(model_result) for model_result in result.models],
    }
    output_path = write_json_artifact("batch_risk_assessment", payload)
    print(f"batch_risk_assessment={output_path}")
    print(payload)


if __name__ == "__main__":
    main()

