from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.data.go_utils import ASPECT_ROOTS, parse_go_obo, propagate_terms

NAMESPACE_TO_ASPECT = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
}


def parse_term_aspect(obo_path):
    term_aspect: dict[str, str] = {}
    current_id = None
    in_term = False
    is_obsolete = False
    namespace = None
    with Path(obo_path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("["):
                if in_term and current_id and not is_obsolete and namespace:
                    aspect = NAMESPACE_TO_ASPECT.get(namespace)
                    if aspect:
                        term_aspect[current_id] = aspect
                in_term = line == "[Term]"
                current_id = None
                namespace = None
                is_obsolete = False
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:]
            elif line.startswith("namespace: "):
                namespace = line[len("namespace: "):]
            elif line == "is_obsolete: true":
                is_obsolete = True
    if in_term and current_id and not is_obsolete and namespace:
        aspect = NAMESPACE_TO_ASPECT.get(namespace)
        if aspect:
            term_aspect[current_id] = aspect
    return term_aspect

SCRIPT_DIR = Path(__file__).resolve().parent
PRED_DIR = SCRIPT_DIR / "predictions"
TEST_TSV = SCRIPT_DIR / "fasta" / "test" / "test.tsv"
TRAIN_TSV = SCRIPT_DIR / "fasta" / "training" / "training.tsv"
OBO_PATH = SCRIPT_DIR / "fasta" / "go-basic.obo"
IC_PATH = SCRIPT_DIR / "fasta" / "ic.pkl"
REPORT_PATH = SCRIPT_DIR / "REPORT.md"
CACHE_PATH = SCRIPT_DIR / "compare_cache.json"

ASPECTS = ("F", "P", "C")
ASPECT_NAME = {"F": "MFO", "P": "BPO", "C": "CCO"}
PRED_ASPECT_MAP = {"f": "F", "p": "P", "c": "C", "F": "F", "P": "P", "C": "C"}
N_THRESHOLDS = 101


def load_ground_truth(parents):
    df = pd.read_csv(TEST_TSV, sep="\t")
    truth = {a: defaultdict(set) for a in ASPECTS}
    for entry, term, aspect in zip(df["EntryID"], df["term"], df["aspect"]):
        if aspect not in ASPECTS:
            continue
        truth[aspect][entry].add(term)
    propagated = {a: {} for a in ASPECTS}
    for aspect in ASPECTS:
        root = ASPECT_ROOTS[aspect]
        for entry, terms in truth[aspect].items():
            full = propagate_terms(terms, parents)
            full.discard(root)
            if full:
                propagated[aspect][entry] = full
    return propagated


def load_training_label_space(term_aspect):
    df = pd.read_csv(TRAIN_TSV, sep="\t")
    eval_terms = {a: set() for a in ASPECTS}
    for term in df["term"]:
        aspect = term_aspect.get(term)
        if aspect in ASPECTS:
            eval_terms[aspect].add(term)
    for aspect in ASPECTS:
        eval_terms[aspect].discard(ASPECT_ROOTS[aspect])
    return eval_terms


def load_prediction(path, parents, term_aspect, eval_terms):
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n").split("\t")
    has_header = not first[0].startswith("GO:") and not first[1].startswith("GO:")
    n_cols = len(first)
    if n_cols == 4:
        names = ["EntryID", "term", "score", "aspect"]
    elif n_cols == 3:
        names = ["EntryID", "term", "score"]
    else:
        raise ValueError(f"Unexpected column count {n_cols} in {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=names,
        dtype={"EntryID": str, "term": str, "score": float, "aspect": str} if n_cols == 4
        else {"EntryID": str, "term": str, "score": float},
    )
    if "aspect" not in df.columns:
        df["aspect"] = df["term"].map(term_aspect)
    else:
        df["aspect"] = df["aspect"].map(PRED_ASPECT_MAP)
    df = df.dropna(subset=["aspect"])
    preds = {a: defaultdict(dict) for a in ASPECTS}
    for aspect in ASPECTS:
        root = ASPECT_ROOTS[aspect]
        sub = df[df["aspect"] == aspect]
        per_protein = defaultdict(list)
        for entry, term, score in zip(sub["EntryID"], sub["term"], sub["score"]):
            per_protein[entry].append((term, score))
        allowed = eval_terms[aspect]
        for entry, items in per_protein.items():
            term_score: dict[str, float] = {}
            for term, score in items:
                ancestors_set = propagate_terms([term], parents)
                for anc in ancestors_set:
                    if anc == root:
                        continue
                    if anc not in allowed:
                        continue
                    if score > term_score.get(anc, -1.0):
                        term_score[anc] = score
            if term_score:
                preds[aspect][entry] = term_score
    return preds


def compute_aspect_metrics(truth_aspect, pred_aspect, ic_aspect):
    proteins = list(truth_aspect.keys())
    n_proteins = len(proteins)
    if n_proteins == 0:
        return {"fmax": float("nan"), "fmax_threshold": float("nan"),
                "smin": float("nan"), "aupr": float("nan"), "auc": float("nan")}

    thresholds = np.linspace(0.0, 1.0, N_THRESHOLDS)
    sum_prec = np.zeros(N_THRESHOLDS)
    cnt_prec = np.zeros(N_THRESHOLDS)
    sum_rec = np.zeros(N_THRESHOLDS)
    sum_ru = np.zeros(N_THRESHOLDS)
    sum_mi = np.zeros(N_THRESHOLDS)

    for entry in proteins:
        true_terms = truth_aspect[entry]
        pred_terms = pred_aspect.get(entry, {})
        true_ic_total = sum(ic_aspect.get(t, 0.0) for t in true_terms)
        items = list(pred_terms.items())
        scores = np.array([s for _, s in items], dtype=float) if items else np.zeros(0)
        terms = [t for t, _ in items]
        ics = np.array([ic_aspect.get(t, 0.0) for t in terms], dtype=float) if items else np.zeros(0)
        is_true = np.array([t in true_terms for t in terms], dtype=bool) if items else np.zeros(0, dtype=bool)
        for i, tau in enumerate(thresholds):
            if items:
                sel = scores >= tau
                n_pred = int(sel.sum())
                tp_ic = float(ics[sel & is_true].sum())
                fp_ic = float(ics[sel & ~is_true].sum())
                tp_count = int((sel & is_true).sum())
            else:
                n_pred = 0
                tp_ic = 0.0
                fp_ic = 0.0
                tp_count = 0
            if n_pred > 0:
                sum_prec[i] += tp_count / n_pred
                cnt_prec[i] += 1
            sum_rec[i] += tp_count / len(true_terms)
            sum_ru[i] += max(true_ic_total - tp_ic, 0.0)
            sum_mi[i] += fp_ic

    avg_prec = np.where(cnt_prec > 0, sum_prec / np.maximum(cnt_prec, 1), 0.0)
    avg_rec = sum_rec / n_proteins
    ru = sum_ru / n_proteins
    mi = sum_mi / n_proteins

    f = np.where((avg_prec + avg_rec) > 0, 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-12), 0.0)
    fmax_idx = int(np.argmax(f))
    fmax = float(f[fmax_idx])
    fmax_threshold = float(thresholds[fmax_idx])

    s = np.sqrt(ru ** 2 + mi ** 2)
    smin = float(np.min(s))

    order = np.argsort(avg_rec)
    rec_sorted = avg_rec[order]
    prec_sorted = avg_prec[order]
    aupr = float(np.trapz(prec_sorted, rec_sorted))

    return {"fmax": fmax, "fmax_threshold": fmax_threshold,
            "smin": smin, "aupr": aupr,
            "_avg_prec": avg_prec, "_avg_rec": avg_rec}


def compute_class_auc(truth_aspect, pred_aspect):
    proteins = sorted(truth_aspect.keys())
    if not proteins:
        return float("nan")
    class_pos = defaultdict(set)
    class_scores = defaultdict(dict)
    for entry in proteins:
        for term in truth_aspect[entry]:
            class_pos[term].add(entry)
        for term, score in pred_aspect.get(entry, {}).items():
            class_scores[term][entry] = score
    aucs = []
    n_proteins = len(proteins)
    for term, pos_set in class_pos.items():
        n_pos = len(pos_set)
        if n_pos == 0 or n_pos == n_proteins:
            continue
        y_true = np.array([1 if p in pos_set else 0 for p in proteins], dtype=int)
        scores_map = class_scores.get(term, {})
        y_score = np.array([scores_map.get(p, 0.0) for p in proteins], dtype=float)
        aucs.append(roc_auc_score(y_true, y_score))
    return float(np.mean(aucs)) if aucs else float("nan")


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare prediction TSV files against held-out labels")
    parser.add_argument("--no-cache", action="store_true", help="Recompute all metrics and refresh the JSON cache")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help="Path to the JSON metrics cache")
    return parser.parse_args()


def strip_metric_arrays(metrics):
    return {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in metrics.items()
        if not key.startswith("_")
    }


def load_cache(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("methods", {})


def write_cache(path: Path, method_results):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "methods": method_results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def render_report(method_results):
    methods = sorted(method_results.keys())
    lines = ["# Comparison Report", ""]
    for aspect in ASPECTS:
        lines.append(f"## {ASPECT_NAME[aspect]} ({aspect})")
        lines.append("")
        lines.append("| Method | Fmax | Smin | AUPR | AvgAUC |")
        lines.append("| --- | --- | --- | --- | --- |")
        for m in methods:
            r = method_results[m][aspect]
            lines.append(f"| {m} | {fmt(r['fmax'])} | {fmt(r['smin'])} | {fmt(r['aupr'])} | {fmt(r['auc'])} |")
        lines.append("")
    lines.append("## Fmax Thresholds")
    lines.append("")
    header = "| Method | " + " | ".join(ASPECT_NAME[a] for a in ASPECTS) + " |"
    sep = "| --- |" + " --- |" * len(ASPECTS)
    lines.append(header)
    lines.append(sep)
    for m in methods:
        cells = [fmt(method_results[m][a]["fmax_threshold"], 2) for a in ASPECTS]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    parents = parse_go_obo(OBO_PATH)
    term_aspect = parse_term_aspect(OBO_PATH)
    with open(IC_PATH, "rb") as f:
        ic_payload = pickle.load(f)
    ic_by_aspect = ic_payload[2]

    truth = load_ground_truth(parents)
    eval_terms = load_training_label_space(term_aspect)
    for aspect in ASPECTS:
        print(f"[compare] {ASPECT_NAME[aspect]} training label space: {len(eval_terms[aspect])} terms")

    pred_files = sorted(PRED_DIR.glob("*.tsv"))
    if not pred_files:
        raise SystemExit(f"No prediction tsv files found under {PRED_DIR}")
    current_methods = {path.stem for path in pred_files}

    method_results = {} if args.no_cache else load_cache(args.cache)
    method_results = {method: result for method, result in method_results.items() if method in current_methods}
    for path in pred_files:
        method = path.stem
        if method in method_results:
            print(f"[compare] using cache for {method}", flush=True)
            continue
        print(f"[compare] processing {method} ...", flush=True)
        preds = load_prediction(path, parents, term_aspect, eval_terms)
        per_aspect = {}
        for aspect in ASPECTS:
            metrics = compute_aspect_metrics(truth[aspect], preds[aspect], ic_by_aspect[aspect])
            metrics["auc"] = compute_class_auc(truth[aspect], preds[aspect])
            per_aspect[aspect] = strip_metric_arrays(metrics)
        method_results[method] = per_aspect

    write_cache(args.cache, method_results)
    render_report(method_results)
    print(f"[compare] wrote cache {args.cache}")
    print(f"[compare] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
