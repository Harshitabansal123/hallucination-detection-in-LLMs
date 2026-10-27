# Claim verification (Person 3)

Takes each claim and its retrieved evidence (Person 2) and labels the claim
**supported / contradicted / unsupported**, with raw scores for the confidence + explanation step (Person 4).

## How it works
1. `src/verification/pipeline.py` reads `data/processed/evidence.json`.
2. Each passage is prefixed with its title and split into sentence-based chunks of at most 800 characters
   (the NLI model only reads about 512 tokens).
3. `src/verification/verifier.py` scores every (chunk, claim) pair with a multilingual NLI model
   (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`): evidence = premise, claim = hypothesis.
   entailment -> supported, contradiction -> contradicted, neutral -> unsupported (evidence insufficient, NOT refutation).
4. Per claim, the strongest entailment and the strongest contradiction over all chunks decide the label (threshold 0.5).

## Run
```
pip install torch transformers sentencepiece
python -m src.verification.pipeline                       # evidence.json -> verification.json
python -m src.verification.pipeline --evidence A.json --out B.json
python -m experiments.inspect_verification 1:2            # show per-passage scores for sample 1, claim 2
```
The first run downloads the model (about 1 GB). 75 claims take several minutes on CPU.

## Output: `data/processed/verification.json` (one record per claim)
```
{
  "sample_id": 1, "claim_id": 2, "claim": "...", "language": "en",
  "retrieval_status": "success",
  "label": "supported" | "contradicted" | "unsupported",
  "support_score": 0.90,          # strongest entailment probability over all chunks
  "contradict_score": 0.88,       # strongest contradiction probability over all chunks
  "best_evidence": { "text", "title", "url", "source_type", "docid", "rank", "chunk", ... },
  "reason": null,                 # "no evidence retrieved" when there was nothing to score
  "num_passages": 10, "num_chunks_scored": 15,
  "passage_scores": [ { "title", "url", "source_type", "chunk", "text_preview",
                        "entailment", "neutral", "contradiction" }, ... ]   # one entry per chunk
}
```
`source_type` is one of `miracl`, `marathi_wikipedia`, `sangraha_verified`, `web_search`.

## Notes for the confidence / explanation step (Person 4)
These come from a first look on a small, DRAFT-labelled set, so treat them as hints, not results.
- The scores are raw softmax probabilities and are over-confident (0.99+ is common). They are not calibrated.
- Evidence often disagrees with itself: on the current 75 claims, about 20 have BOTH a support and a
  contradiction score of 0.8 or more from different chunks. A good confidence value should drop in that case.
  `passage_scores` lets you count how many chunks support vs contradict, and from which sources.
- Taking the single max over many chunks is noisy (about 17 chunks per claim on average).
  Averaging the top 3 looked slightly better but the difference is within noise.
- The label "unsupported" is currently never produced at threshold 0.5, because some chunk almost always scores
  above it. Do not read "no unsupported results" as "everything is supported".
- Web-search evidence reduced false alarms in most languages. Keep `source_type` when reporting results.
- Garbled claims (see Hindi samples 5 and 6) can still be labelled supported through word overlap.

## Evaluation tools (`experiments/`)
- `score_verification.py`: scores `verification.json` against hand labels and compares decision rules
  and evidence sources per language, without re-running the model (`--split dev|test` is supported).
- `evaluate_verification.py` + `adapters/fever.py`, `adapters/xnli.py`: label-based evaluation on public datasets.
- `check_claims_dataset.py`: sanity checks on a claims file.
- `data/eval/`: draft gold labels and a draft false-claim set (en/hi/bn/mr).
  **All labels and the Hindi/Bengali/Marathi translations are unreviewed drafts** (see `review_sheet.csv`).

## Known limitations
- Only 6 false/nonsense claims in the current labelled data, none in English or Marathi.
- Retrieval is same-language only, so cross-lingual verification is not exercised yet.
- Bengali and Marathi quality of the NLI model has not been measured separately.
