import json
import shutil
from pathlib import Path
from datetime import datetime

path = Path("data/processed/claims.json")

if not path.exists():
    raise SystemExit(f"File not found: {path.resolve()}")

data = json.loads(path.read_text(encoding="utf-8"))
samples = data["samples"] if isinstance(data, dict) else data

sample = next(
    (s for s in samples if str(s.get("sample_id")) == "1"),
    None
)

if sample is None:
    raise SystemExit("Sample ID 1 was not found.")

if sample.get("language") == "en":
    print("Sample 1 is already labelled English. No change needed.")
    raise SystemExit(0)

if sample.get("language") not in (None, ""):
    raise SystemExit(
        f"Sample 1 already has language={sample.get('language')!r}; "
        "inspect it before changing anything."
    )

question = sample.get("question", "")
answer = sample.get("answer", "")

if question.strip().lower() != "who discovered penicillin?":
    raise SystemExit("Unexpected question for Sample 1; no changes made.")

if "Alexander Fleming" not in answer or "penicillin" not in answer.lower():
    raise SystemExit("Sample 1 answer did not pass the expected-content check.")

backup = path.with_name(
    f"claims.before_language_fix.{datetime.now():%Y%m%d_%H%M%S}.json"
)
shutil.copy2(path, backup)

sample["language"] = "en"

tmp = path.with_suffix(".json.tmp")
tmp.write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)
tmp.replace(path)

print(f"Added language='en' to Sample 1.")
print(f"Backup created: {backup}")