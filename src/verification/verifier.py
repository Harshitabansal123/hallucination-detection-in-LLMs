"""
Person 3: Claim Verification

Input : one claim + a list of retrieved evidence passages (from Person 2)
Output: Supported / Contradicted / Unsupported + raw scores (for Person 4)

Approach: multilingual NLI.
    premise    = evidence passage
    hypothesis = claim
    entailment    -> supported
    contradiction -> contradicted
    neutral       -> unsupported (evidence insufficient, NOT refutation)

Claim and evidence may be in different languages, because the NLI model
is multilingual.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Decision thresholds. Tune these on validation data.
SUPPORT_THRESHOLD = 0.5
CONTRADICT_THRESHOLD = 0.5

_tokenizer = None
_model = None


def _load_model():
    """Lazy-load so importing this file does not download/load a model."""
    global _tokenizer, _model

    if _model is None:
        print("Loading NLI model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.to("cuda")
        print("NLI model loaded.")

    return _tokenizer, _model


def nli_scores(premise: str, hypothesis: str) -> dict:
    """
    Run NLI on one (evidence, claim) pair.
    Returns probabilities: {"entailment": p, "neutral": p, "contradiction": p}
    """
    tokenizer, model = _load_model()

    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=-1).tolist()

    # Read label names from the model config instead of assuming an order.
    scores = {}
    for idx, p in enumerate(probs):
        scores[model.config.id2label[idx].lower()] = p

    return scores


def aggregate(
    passage_results: list[dict],
    support_threshold: float = SUPPORT_THRESHOLD,
    contradict_threshold: float = CONTRADICT_THRESHOLD,
) -> dict:
    """
    Combine per-passage NLI scores into one decision for the claim.

    Rule:
      - best entailment >= support_threshold and >= best contradiction -> supported
      - else best contradiction >= contradict_threshold                 -> contradicted
      - else                                                            -> unsupported
    Pure function (no model), so it is easy to test.
    """
    if not passage_results:
        return {
            "label": "unsupported",
            "support_score": 0.0,
            "contradict_score": 0.0,
            "best_evidence": None,
            "reason": "no evidence retrieved",
        }

    best_sup = max(passage_results, key=lambda r: r["scores"]["entailment"])
    best_con = max(passage_results, key=lambda r: r["scores"]["contradiction"])

    sup = best_sup["scores"]["entailment"]
    con = best_con["scores"]["contradiction"]

    if sup >= support_threshold and sup >= con:
        label, best = "supported", best_sup
    elif con >= contradict_threshold:
        label, best = "contradicted", best_con
    else:
        label, best = "unsupported", best_sup

    return {
        "label": label,
        "support_score": sup,
        "contradict_score": con,
        "best_evidence": best["evidence"],
    }


def verify_claim(claim: str, evidence: list[dict]) -> dict:
    """
    claim   : claim text
    evidence: [{"text": "...", "source": "...", "language": "en"}, ...]
    """
    passage_results = []

    for ev in evidence:
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        passage_results.append(
            {"evidence": ev, "scores": nli_scores(text, claim)}
        )

    result = aggregate(passage_results)
    result["claim"] = claim
    result["num_passages"] = len(passage_results)

    return result


def verify_sample(sample: dict) -> dict:
    """
    sample: one entry like those in claims.json, where each claim also has
            an "evidence" list: {"claim_id": 1, "claim": "...", "evidence": [...]}
    Returns the sample with a "verification" list added.
    """
    verification = []

    for item in sample.get("claims", []):
        res = verify_claim(item["claim"], item.get("evidence", []))
        res["claim_id"] = item["claim_id"]
        verification.append(res)

    out = dict(sample)
    out["verification"] = verification
    return out
