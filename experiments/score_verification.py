"""
Score data/processed/verification.json against hand labels, and test other ways of
combining the per-chunk NLI scores WITHOUT re-running the model.

Run from the repo root:
    python -m experiments.score_verification
    python -m experiments.score_verification data/processed/verification.json data/eval/gold_claim_labels.json

Read-only: it never writes any file.

IMPORTANT: the labels shipped in gold_claim_labels.json are a DRAFT and the dataset has only a handful
of false claims, so every number here is noisy. Use this to see PATTERNS, not to report final results.
"""

import json
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

LABELS = ["supported", "contradicted", "unsupported"]
BAD = {"contradicted", "unsupported"}          # a claim a user should be warned about


# ---------------------------------------------------------------- decision rules
def decide(sup, con, t, m):
    """Supported/contradicted need score >= t AND to beat the other score by margin m; else unsupported."""
    if sup >= t and sup - con >= m:
        return "supported"
    if con >= t and con - sup >= m:
        return "contradicted"
    return "unsupported"


def scores_for(rule, ent, con):
    if rule == "max":
        return max(ent), max(con)
    if rule == "top3":
        top = lambda xs: sum(sorted(xs, reverse=True)[:3]) / min(3, len(xs))
        return top(ent), top(con)
    raise ValueError(rule)


# ---------------------------------------------------------------- metrics
def metrics(gold, pred):
    conf = {g: Counter() for g in LABELS}
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    f1s = []
    for lab in LABELS:
        tp = conf[lab][lab]
        fp = sum(conf[g][lab] for g in LABELS if g != lab)
        fn = sum(conf[lab][p] for p in LABELS if p != lab)
        if tp + fn == 0:
            continue                                      # label absent from gold: skip in macro-F1
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    bad_gold = [(g, p) for g, p in zip(gold, pred) if g in BAD]
    ok_gold = [(g, p) for g, p in zip(gold, pred) if g == "supported"]
    catch = sum(p in BAD for _, p in bad_gold) / len(bad_gold) if bad_gold else float("nan")
    false_alarm = sum(p in BAD for _, p in ok_gold) / len(ok_gold) if ok_gold else float("nan")
    return {
        "n": len(gold), "acc": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "catch": catch, "false_alarm": false_alarm,
        "bad_n": len(bad_gold), "ok_n": len(ok_gold), "conf": conf,
    }


# ---------------------------------------------------------------- per-language / per-source breakdown
def frac(k, n):
    """'k/n', or 'n/a' when the denominator is zero (no such claims) - never a misleading percentage."""
    return "n/a" if n == 0 else f"{k}/{n}"


def lang_stats(rows, rule, t=0.5):
    """Counts per language for one decision rule. rows come from build_rows(); no model needed."""
    out = {}
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        preds = [decide(*scores_for(rule, r["ent"], r["con"]), t, 0.0) for r in sub]
        bad = [(r, p) for r, p in zip(sub, preds) if r["gold"] in BAD]
        ok = [(r, p) for r, p in zip(sub, preds) if r["gold"] == "supported"]
        out[lang] = {
            "n": len(sub),
            "caught": sum(p in BAD for _, p in bad), "bad_n": len(bad),
            "fa": sum(p in BAD for _, p in ok), "ok_n": len(ok),
            "strong": sum(r["max_c"] >= 0.8 for r, _ in ok),
            "noev": sum(r["n_chunks"] == 0 for r in sub),
        }
    return out


def print_language_source_breakdown(ver, build_rows):
    """For each language: results with each evidence source, then 'does adding web help?' per language."""
    web = lambda p: p.get("source_type") == "web_search"
    corpus = lambda p: p.get("source_type") != "web_search"
    langs = sorted({r["language"] for r in ver})

    print("\n=== per-language x per-source breakdown (thresholds unchanged: 0.5) ===")
    print("Cells are counts (k/n); 'n/a' = no such claims. 'no-chunk' = claims left with zero chunks from that source")
    print("(they get 'unsupported', which counts as flagged). strong-con = TRUE claims with a chunk contradiction >= 0.8.")

    summary = {}                                                  # (lang, rule) -> {"corpus": stats, "all": stats}
    for lang in langs:
        src_here = sorted({p.get("source_type") or "unknown" for r in ver if r["language"] == lang for p in r["passage_scores"]})
        corpus_src = [x for x in src_here if x != "web_search"]
        subsets = [("all sources", lambda p: True)]
        if len(corpus_src) > 1:
            subsets.append(("corpus only", corpus))
        for x in src_here:
            subsets.append((f"{x} only", (lambda p, x=x: (p.get("source_type") or "unknown") == x)))

        base = lang_stats(build_rows(lambda p: True), "max").get(lang)
        if base is None:
            continue
        print(f"\n--- {lang}: {base['n']} claims | true {base['ok_n']} | false/nonsense {base['bad_n']} ---")
        print(f"{'kept chunks':26s} {'rule':5s} | {'no-chunk':8s} | {'caught':7s} | {'false alarms':12s} | strong-con")
        for name, keep in subsets:
            rows = build_rows(keep)
            for rule in ("max", "top3"):
                st = lang_stats(rows, rule).get(lang)
                if st is None:
                    continue
                print(f"{name:26s} {rule:5s} | {st['noev']:>8d} | {frac(st['caught'], st['bad_n']):>7s} | "
                      f"{frac(st['fa'], st['ok_n']):>12s} | {frac(st['strong'], st['ok_n'])}")
        for rule in ("max", "top3"):
            summary[(lang, rule)] = {
                "all": lang_stats(build_rows(lambda p: True), rule).get(lang),
                "corpus": lang_stats(build_rows(corpus), rule).get(lang),
                "web": lang_stats(build_rows(web), rule).get(lang),
            }

    print("\n=== does adding web evidence change the results, per language? ===")
    print("(true claims wrongly flagged  |  false claims caught)   corpus only -> all sources")
    for rule in ("max", "top3"):
        print(f"\nrule: {rule}")
        print(f"{'lang':5s} | {'false alarms':^24s} | {'caught':^22s} | web-only false alarms")
        for lang in langs:
            e = summary.get((lang, rule))
            if not e or not e["all"]:
                continue
            a, c, w = e["all"], e["corpus"], e["web"]
            print(f"{lang:5s} | {frac(c['fa'], c['ok_n']):>9s} -> {frac(a['fa'], a['ok_n']):<9s}      | "
                  f"{frac(c['caught'], c['bad_n']):>7s} -> {frac(a['caught'], a['bad_n']):<7s}     | {frac(w['fa'], w['ok_n'])}")
    print("\nHow to read it: a language where corpus-only already has few false alarms shows little web effect;")
    print("a cell with n <= 5 can change by one claim. Languages with 'n/a' under 'caught' cannot say anything")
    print("about detection at all - they need false claims in the test set first.")


def main():
    args = sys.argv[1:]
    split = None                                   # optional: --split dev | test
    if "--split" in args:
        i = args.index("--split")
        split = args[i + 1]
        del args[i:i + 2]
    ver_path = args[0] if len(args) > 0 else "data/processed/verification.json"
    gold_path = args[1] if len(args) > 1 else "data/eval/gold_claim_labels.json"
    ver = json.load(open(ver_path, encoding="utf-8"))
    labels = json.load(open(gold_path, encoding="utf-8"))["labels"]
    if split:
        labels = [g for g in labels if g.get("split") == split]
        print(f"[split filter] keeping only '{split}' claims: {len(labels)} labelled")
        if not labels:
            sys.exit("No labels with that split. Does the gold file have a 'split' field?")
    gold = {(g["sample_id"], g["claim_id"]): g for g in labels}

    def build_rows(keep=lambda p: True):
        out = []
        for r in ver:
            g = gold.get((r["sample_id"], r["claim_id"]))
            if g is None:
                continue
            chunks = [p for p in r["passage_scores"] if keep(p)]
            ent = [p["entailment"] for p in chunks] or [0.0]
            con = [p["contradiction"] for p in chunks] or [0.0]
            out.append({"id": f"{r['sample_id']}:{r['claim_id']}", "lang": r["language"], "gold": g["label"],
                        "claim": r["claim"], "pred": r["label"], "ent": ent, "con": con,
                        "max_e": max(ent), "max_c": max(con), "n_chunks": len(chunks)})
        return out

    rows = build_rows()
    missing = len(ver) - len(rows)
    print(f"Scored {len(rows)} claims" + (f" ({missing} had no gold label)" if missing else ""))

    # ---- 1. the current pipeline output
    gold_l, pred_l = [r["gold"] for r in rows], [r["pred"] for r in rows]
    m = metrics(gold_l, pred_l)
    print(f"\n=== CURRENT pipeline (max over chunks, threshold 0.5) ===")
    print(f"accuracy {m['acc']:.2f} | macro-F1 {m['macro_f1']:.2f}")
    print(f"false/nonsense claims caught: {m['catch']:.0%} of {m['bad_n']}   |   true claims wrongly flagged: {m['false_alarm']:.0%} of {m['ok_n']}")
    print("predictions never used:", [l for l in LABELS if l not in pred_l] or "none")
    print("\nconfusion (rows = my label, cols = pipeline)")
    print(" " * 14 + "".join(f"{l[:6]:>9s}" for l in LABELS))
    for g in LABELS:
        print(f"{g:14s}" + "".join(f"{m['conf'][g][p]:9d}" for p in LABELS))

    print("\n--- by language ---")
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    for lang, rs in sorted(by_lang.items()):
        mm = metrics([r["gold"] for r in rs], [r["pred"] for r in rs])
        print(f"  {lang}: n={mm['n']:2d}  acc {mm['acc']:.2f}  caught {mm['catch']:.0%} of {mm['bad_n']}  false alarms {mm['false_alarm']:.0%} of {mm['ok_n']}")

    print("\n--- disagreements (my label vs pipeline) ---")
    for r in rows:
        if r["gold"] != r["pred"]:
            print(f"  {r['id']:>6} [{r['lang']}] mine={r['gold']:<12} pipeline={r['pred']:<12} "
                  f"maxEnt={r['max_e']:.2f} maxCon={r['max_c']:.2f} chunks={r['n_chunks']}  | {r['claim'][:55]}")

    # ---- 2. how often are BOTH scores high? (a sign that max-over-chunks is picking up noise)
    both = [r for r in rows if r["max_e"] >= 0.8 and r["max_c"] >= 0.8]
    high_c_true = [r for r in rows if r["max_c"] >= 0.8 and r["gold"] == "supported"]
    print(f"\n=== conflicting evidence ===")
    print(f"claims where max entailment AND max contradiction are both >= 0.8: {len(both)} of {len(rows)}")
    print(f"TRUE claims with a contradiction score >= 0.8 from at least one chunk: {len(high_c_true)} of {m['ok_n']}")
    print("average chunks scored per claim:", f"{sum(r['n_chunks'] for r in rows) / len(rows):.1f}",
          "(more chunks = more chances for a spurious high score)")

    # ---- 3. other ways to combine the same chunk scores (no model re-run)
    results = []
    for rule in ("max", "top3"):
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            for mg in (0.0, 0.2, 0.4, 0.6):
                pred = [decide(*scores_for(rule, r["ent"], r["con"]), t, mg) for r in rows]
                mm = metrics(gold_l, pred)
                results.append((rule, t, mg, mm, set(pred)))
    results.sort(key=lambda x: ((x[3]["catch"] + (1 - x[3]["false_alarm"])) / 2, x[3]["macro_f1"]), reverse=True)
    print("\n=== other decision rules on the SAME chunk scores (best 10 by balanced catch / false-alarm) ===")
    print("rule  thresh margin | acc  macroF1 | caught  false-alarms | labels used")
    for rule, t, mg, mm, used in results[:10]:
        print(f"{rule:5s} {t:5.2f} {mg:6.1f} | {mm['acc']:.2f}  {mm['macro_f1']:.2f}   | {mm['catch']:5.0%}   {mm['false_alarm']:6.0%}       | {sorted(used)}")
    # ---- 4. does web-search evidence help or add noise? (same chunk scores, different subsets)
    sources = sorted({p.get("source_type") or "unknown" for r in ver for p in r["passage_scores"]})
    print("\n=== evidence source ablation (same model scores, different chunks kept) ===")
    print("chunks by source:", {src: sum(1 for r in ver for p in r["passage_scores"] if (p.get('source_type') or 'unknown') == src) for src in sources})
    print("kept chunks            rule        | acc  | caught  false-alarms | true claims with a chunk contradiction >= 0.8")
    subsets = [("all sources", lambda p: True),
               ("corpus only (no web)", lambda p: p.get("source_type") != "web_search"),
               ("web search only", lambda p: p.get("source_type") == "web_search")]
    for name, keep in subsets:
        rs = build_rows(keep)
        g_l = [r["gold"] for r in rs]
        hi_con = sum(1 for r in rs if r["gold"] == "supported" and r["max_c"] >= 0.8)
        for rule in ("max", "top3"):
            pred = [decide(*scores_for(rule, r["ent"], r["con"]), 0.5, 0.0) for r in rs]
            mm = metrics(g_l, pred)
            print(f"{name:22s} {rule:5s} t=0.5 | {mm['acc']:.2f} | {mm['catch']:5.0%}   {mm['false_alarm']:6.0%}       | {hi_con} of {mm['ok_n']}")
    print("(compare the rows: a clear drop in false alarms with web included means web evidence helps; a rise means noise)")

    print_language_source_breakdown(ver, build_rows)

    print("\nCaution: with only a few false claims, tuning on this file overfits. Treat it as a hint about which")
    print("KIND of rule helps (margin? averaging?), then confirm on a larger, human-checked set.")


if __name__ == "__main__":
    main()
