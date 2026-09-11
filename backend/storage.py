import json

import os

from pathlib import Path

from typing import Any, Dict, List

ARTIFACT_ROOT = Path(

    os.getenv("AUREXIS_ARTIFACT_DIR", "./artifacts")

)

MODEL_CARD_DIR = ARTIFACT_ROOT / "model_cards"

REPORT_DIR = ARTIFACT_ROOT / "reports"

LOG_FILE = ARTIFACT_ROOT / "audit_log.jsonl"

MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

def safe_json_default(value: Any) -> Any:

    """

    Convert NumPy/Pandas-style objects into JSON-safe values.

    """

    if hasattr(value, "item"):

        return value.item()

    if hasattr(value, "tolist"):

        return value.tolist()

    return str(value)

def write_audit_event(record: Dict[str, Any]) -> None:

    """

    Append one governance event to the JSONL audit trail.

    """

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:

        file.write(

            json.dumps(

                record,

                default=safe_json_default,

                ensure_ascii=False,

            )

            + "\n"

        )

def read_audit_events() -> List[Dict[str, Any]]:

    """

    Read all valid governance events.

    """

    if not LOG_FILE.exists():

        return []

    records: List[Dict[str, Any]] = []

    with LOG_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:

                continue

            try:

                records.append(json.loads(line))

            except json.JSONDecodeError:

                continue

    return records

def save_model_card(model_card: Dict[str, Any]) -> Path:

    """

    Save a model card as JSON.

    """

    model_id = str(

        model_card.get("model_id", "unknown_model")

    )

    path = MODEL_CARD_DIR / f"{model_id}.json"

    with path.open("w", encoding="utf-8") as file:

        json.dump(

            model_card,

            file,

            indent=2,

            ensure_ascii=False,

            default=safe_json_default,

        )

    return path

def load_model_cards() -> List[Dict[str, Any]]:

    """

    Load all model cards from the repository.

    """

    if not MODEL_CARD_DIR.exists():

        return []

    cards: List[Dict[str, Any]] = []

    for path in sorted(MODEL_CARD_DIR.glob("*.json")):

        try:

            with path.open("r", encoding="utf-8") as file:

                cards.append(json.load(file))

        except (json.JSONDecodeError, OSError):

            continue

    return cards
