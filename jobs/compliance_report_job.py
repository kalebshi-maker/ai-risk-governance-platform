"""Nebius Job #4: Compliance Report Generator.

Generates jurisdiction-specific compliance evidence reports from the latest
autonomous governance run.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from aurexis_twin import AuditStore, generate_compliance_report, run_governance_cycle, write_json_artifact


JURISDICTIONS = ["EU AI Act", "SR 11-7", "UK Model Risk Guidance"]


def main() -> None:
    audit = AuditStore()
    reports = []
    for jurisdiction in JURISDICTIONS:
        result = run_governance_cycle(model_name="scheduled_rf", jurisdiction=jurisdiction, audit_store=audit)
        report_path = generate_compliance_report(result, audit.read(limit=100), jurisdiction=jurisdiction)
        reports.append({"jurisdiction": jurisdiction, "result": asdict(result), "report_path": str(report_path)})

    payload = {"reports": reports}
    output_path = write_json_artifact("compliance_reports", payload)
    print(f"compliance_reports={output_path}")
    print(payload)


if __name__ == "__main__":
    main()

