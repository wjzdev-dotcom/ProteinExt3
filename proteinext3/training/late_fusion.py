from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from training.data.go_utils import build_propagation_indices, parse_go_obo, propagate_scores
from training.train import DEFAULT_IC_PATH, load_information_content
from training.trainer import compute_multilabel_metrics

OOF_DIR = ROOT_DIR / "training" / "oof"
OUTPUT_PATH = ROOT_DIR / "models_raw" / "latefusion_new.csv"
OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"
DEFAULT_METHODS = ("esm2-33", "esm2-28", "esm2-20", "prott5", "blast")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
METHOD_COLUMNS = {"esm2-33": "l33", "esm2-28": "l28", "esm2-20": "l20", "prott5": "t5", "blast": "blast"}
THRESHOLDS = np.linspace(0.01, 0.99, 99, dtype=np.float64)
T_BINS = len(THRESHOLDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search OOF late-fusion weights by fused FMAX")
    parser.add_argument("--aspect", nargs="+", default=["P", "F", "C"], choices=["P", "F", "C"])
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="output CSV path")
    parser.add_argument("--propagate", action="store_true", help="Optimize after GO score/label propagation")
    parser.add_argument("--ic-pkl", type=Path, default=DEFAULT_IC_PATH)
    parser.add_argument("--fmax-alpha", type=float, default=1.0, help="Weight for Fmax in fusion search score")
    parser.add_argument("--aupr-beta", type=float, default=0.0, help="Weight for AUPR in fusion search score")
    parser.add_argument("--blast-max-weight", type=float, default=1.0, help="Maximum allowed BLAST fusion weight")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but CUDA is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("Requested mps but MPS is not available")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def method_column(method: str) -> str:
    return METHOD_COLUMNS.get(method, method.replace("-", "_"))


def discover_methods(aspect: str) -> List[str]:
    """Methods that have all DEFAULT_FOLDS available for this aspect."""
    marker = f"_{aspect}_"
    per_fold: List[set] = []
    for fold in DEFAULT_FOLDS:
        found = set()
        for path in OOF_DIR.glob(f"**/*_{aspect}_*fold_{fold}.npz"):
            if marker not in path.stem:
                continue
            found.add(path.stem.split(marker, 1)[0])
        per_fold.append(found)
    available = set.intersection(*per_fold) if per_fold else set()
    preferred = [m for m in DEFAULT_METHODS if m in available]
    extras = sorted(available - set(preferred))
    methods = preferred + extras
    if not methods:
        raise RuntimeError(f"No method has all folds {list(DEFAULT_FOLDS)} for aspect={aspect}")
    return methods


def find_oof_path(method: str, aspect: str, fold: int) -> Path:
    exact = OOF_DIR / method / f"{method}_{aspect}_fold_{fold}.npz"
    if exact.exists():
        return exact
    marker = f"_{aspect}_"
    matches = sorted(
        p for p in OOF_DIR.glob(f"**/{method}_{aspect}_*fold_{fold}.npz")
        if marker in p.stem and p.stem.split(marker, 1)[0] == method
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"OOF for method={method} aspect={aspect} fold={fold}: {matches}")
    return matches[0]


def load_npz(method: str, aspect: str, fold: int) -> np.lib.npyio.NpzFile:
    return np.load(find_oof_path(method, aspect, fold), allow_pickle=True)


def validate_step(step: float) -> None:
    if step <= 0 or step > 1:
        raise ValueError(f"step must be in (0, 1], got {step}")
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0, atol=1e-8):
        raise ValueError(f"step must divide 1.0 exactly, got {step}")


def simplex_grid(n_methods: int, step: float) -> np.ndarray:
    validate_step(step)
    units = int(round(1.0 / step))

    def points(n: int, total: int):
        if n == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in points(n - 1, total - first):
                yield (first,) + rest

    return np.array(list(points(n_methods, units)), dtype=np.float32) * step


def neighborhood_grid(centers: np.ndarray, step: float, radius: float) -> np.ndarray:
    validate_step(step)
    units = int(round(1.0 / step))
    n = centers.shape[1]
    radius_units = int(round(radius / step + 1e-9))
    seen: set = set()
    out: List[tuple] = []
    for center in centers:
        ctr = np.round(center / step).astype(int)
        bounds = [(max(0, c - radius_units), min(units, c + radius_units)) for c in ctr]

        def enum(idx: int, remaining: int):
            lo, hi = bounds[idx]
            if idx == n - 1:
                if lo <= remaining <= hi:
                    yield (remaining,)
                return
            for v in range(max(lo, 0), min(hi, remaining) + 1):
                yield from ((v,) + r for r in enum(idx + 1, remaining - v))

        for pt in enum(0, units):
            if pt not in seen:
                seen.add(pt)
                out.append(pt)
    return np.array(out, dtype=np.float32) * step


def align_matrix(matrix: np.ndarray, src_pids: np.ndarray, src_cls: np.ndarray,
                  tgt_pids: np.ndarray, tgt_cls: np.ndarray) -> np.ndarray:
    if not np.array_equal(src_cls, tgt_cls):
        idx = {str(c): i for i, c in enumerate(src_cls)}
        cols = np.array([idx.get(str(c), -1) for c in tgt_cls])
        out = np.zeros((matrix.shape[0], len(tgt_cls)), dtype=np.float32)
        valid = cols >= 0
        out[:, valid] = matrix[:, cols[valid]].astype(np.float32, copy=False)
        matrix = out
    else:
        matrix = matrix.astype(np.float32, copy=False)
    if not np.array_equal(src_pids, tgt_pids):
        idx = {str(p): i for i, p in enumerate(src_pids)}
        rows = np.array([idx.get(str(p), -1) for p in tgt_pids])
        out = np.zeros((len(tgt_pids), matrix.shape[1]), dtype=np.float32)
        valid = rows >= 0
        out[valid] = matrix[rows[valid]]
        matrix = out
    return matrix


def common_pids(methods: Sequence[str], aspect: str, fold: int) -> np.ndarray:
    sets = [set(str(p) for p in load_npz(m, aspect, fold)["pids"]) for m in methods]
    common = set.intersection(*sets)
    first = load_npz(methods[0], aspect, fold)["pids"]
    return np.asarray([str(p) for p in first if str(p) in common], dtype=object)


def union_classes(methods: Sequence[str], aspect: str) -> np.ndarray:
    terms: set = set()
    for f in DEFAULT_FOLDS:
        for m in methods:
            terms.update(str(t) for t in load_npz(m, aspect, f)["classes"])
    return np.asarray(sorted(terms), dtype=object)


def best_fold(method: str, aspect: str) -> int:
    best = (-1.0, -1)
    for f in DEFAULT_FOLDS:
        m = json.loads(str(load_npz(method, aspect, f)["metrics_json"]))
        if m["fmax"] > best[0]:
            best = (m["fmax"], f)
    return best[1]


def load_fold_cpu(methods: Sequence[str], aspect: str, fold: int, classes: np.ndarray,
                   prop_indices: list | None) -> dict:
    pids = common_pids(methods, aspect, fold)
    M = len(methods)
    P = np.empty((M, len(pids), len(classes)), dtype=np.float32)
    for i, m in enumerate(methods):
        npz = load_npz(m, aspect, fold)
        P[i] = align_matrix(npz["probs"], npz["pids"], npz["classes"], pids, classes)
    npz0 = load_npz(methods[0], aspect, fold)
    raw_labels = align_matrix(npz0["labels"].astype(np.float32, copy=False),
                               npz0["pids"], npz0["classes"], pids, classes)
    labels = (
        propagate_scores(raw_labels, prop_indices)
        if prop_indices is not None else raw_labels
    ).astype(bool, copy=False)
    return {
        "fold": fold,
        "pids": pids,
        "P": P,
        "labels": labels,
        "true_per": labels.sum(axis=1).astype(np.int64),
    }


def torch_prop_indices(prop_indices, device):
    return [torch.as_tensor(c, device=device, dtype=torch.long) if c else None for c in prop_indices]


def propagate_torch_(scores: torch.Tensor, prop_indices_t) -> torch.Tensor:
    """In-place propagation along last dim. scores shape (..., n_cls)."""
    for i, children in enumerate(prop_indices_t):
        if children is None:
            continue
        child_max = torch.index_select(scores, -1, children).amax(dim=-1)
        scores[..., i] = torch.maximum(scores[..., i], child_max)
    return scores


def fmax_metrics_batched(probs: torch.Tensor, labels: torch.Tensor, true_per: torch.Tensor,
                          has_label: torch.Tensor, thresholds: torch.Tensor,
                          class_chunk: int) -> dict:
    """Vectorized FMAX over candidate batch via histogram + reverse cumsum."""
    device = probs.device
    B, n_prot, n_cls = probs.shape
    T = thresholds.shape[0]
    metric_dtype = torch.float32 if device.type == "mps" else torch.float64

    H_total = torch.zeros((B, n_prot, T + 1), dtype=torch.int32, device=device)
    H_pos = torch.zeros((B, n_prot, T + 1), dtype=torch.int32, device=device)
    default_tp = torch.zeros(B, dtype=torch.int64, device=device)
    default_pred_pos = torch.zeros(B, dtype=torch.int64, device=device)
    labels_int = labels.to(torch.int32)
    total_pos = torch.zeros((B, T), dtype=torch.int64, device=device)
    tp_total = torch.zeros((B, T), dtype=torch.int64, device=device)

    for c0 in range(0, n_cls, class_chunk):
        c1 = min(c0 + class_chunk, n_cls)
        chunk = probs[:, :, c0:c1].contiguous()
        bin_idx = torch.searchsorted(thresholds, chunk, right=True).to(torch.int64)
        ones = torch.ones_like(bin_idx, dtype=torch.int32)
        H_total.scatter_add_(2, bin_idx, ones)
        pos_chunk = labels_int[None, :, c0:c1].expand(B, -1, -1).contiguous()
        H_pos.scatter_add_(2, bin_idx, pos_chunk)

        pred05 = chunk >= 0.5
        default_pred_pos += pred05.sum(dim=(1, 2)).to(torch.int64)
        default_tp += torch.logical_and(pred05, labels[None, :, c0:c1]).sum(dim=(1, 2)).to(torch.int64)
        del chunk, bin_idx, ones, pos_chunk, pred05

    pred_per = H_total[:, :, 1:].flip(2).cumsum(dim=2).flip(2)
    tp_per = H_pos[:, :, 1:].flip(2).cumsum(dim=2).flip(2)
    del H_total, H_pos

    has_pred = pred_per > 0
    mask = has_pred & has_label[None, :, None]
    safe = pred_per.clamp(min=1).to(metric_dtype)
    prec = tp_per.to(metric_dtype) / safe
    precision_sum = torch.where(mask, prec, torch.zeros_like(prec)).sum(dim=1)
    precision_count = mask.sum(dim=1)

    true_per_safe = true_per.clamp(min=1).to(metric_dtype)
    rec = tp_per.to(metric_dtype) / true_per_safe[None, :, None]
    rec = torch.where(has_label[None, :, None], rec, torch.zeros_like(rec))
    recall_sum = rec.sum(dim=1)
    total_pos += pred_per.sum(dim=1).to(torch.int64)
    tp_total += tp_per.sum(dim=1).to(torch.int64)

    return {
        "precision_sum": precision_sum,
        "precision_count": precision_count,
        "recall_sum": recall_sum,
        "default_tp": default_tp,
        "default_pred_pos": default_pred_pos,
        "total_pos": total_pos,
        "tp_total": tp_total,
    }


def cand_batch_size(n_cand: int, n_prot: int, n_cls: int, M: int, device: torch.device) -> int:
    bytes_per_cand = n_prot * n_cls * 4
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        budget = int(free * 0.45) - M * bytes_per_cand
    elif device.type == "mps":
        budget = 2_500_000_000 - M * bytes_per_cand
    else:
        budget = 400_000_000
    budget = max(budget, bytes_per_cand)
    return max(1, min(n_cand, budget // max(bytes_per_cand, 1)))


def class_chunk_size(B: int, n_prot: int, device: torch.device) -> int:
    bytes_per = max(1, B * n_prot * 16)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        budget = int(free * 0.2)
    elif device.type == "mps":
        budget = 600_000_000
    else:
        budget = 200_000_000
    return max(1, budget // bytes_per)


def evaluate_candidates(W: np.ndarray, fold_data: List[dict], prop_indices_t,
                         thresholds_t: torch.Tensor, device: torch.device,
                         desc: str, position: int = 0) -> dict:
    K, M = W.shape
    metric_dtype = torch.float32 if device.type == "mps" else torch.float64
    W_t = torch.from_numpy(W).to(device)

    precision_sum = torch.zeros((K, T_BINS), dtype=metric_dtype, device=device)
    precision_count = torch.zeros((K, T_BINS), dtype=torch.int64, device=device)
    recall_sum = torch.zeros((K, T_BINS), dtype=metric_dtype, device=device)
    recall_count = 0
    default_tp = torch.zeros(K, dtype=torch.int64, device=device)
    default_pred_pos = torch.zeros(K, dtype=torch.int64, device=device)
    total_pos = torch.zeros((K, T_BINS), dtype=torch.int64, device=device)
    tp_total = torch.zeros((K, T_BINS), dtype=torch.int64, device=device)
    true_pos_total = 0

    max_n_prot = max(fd["P"].shape[1] for fd in fold_data)
    n_cls = fold_data[0]["P"].shape[2]
    batch = cand_batch_size(K, max_n_prot, n_cls, M, device)

    progress = tqdm(total=K * len(fold_data), desc=desc, position=position, dynamic_ncols=True)
    try:
        for fd in fold_data:
            P = torch.from_numpy(fd["P"]).to(device)
            labels = torch.from_numpy(fd["labels"]).to(device)
            true_per = torch.from_numpy(fd["true_per"]).to(device)
            _, n_prot, fold_n_cls = P.shape
            has_label = true_per > 0
            recall_count += int(has_label.sum().item())
            true_pos_total += int(true_per.sum().item())
            P_flat = P.view(M, -1)

            for c0 in range(0, K, batch):
                c1 = min(c0 + batch, K)
                Wb = W_t[c0:c1]
                fused = (Wb @ P_flat).reshape(c1 - c0, n_prot, fold_n_cls)
                if prop_indices_t is not None:
                    fused = propagate_torch_(fused, prop_indices_t)

                cc = min(fold_n_cls, class_chunk_size(c1 - c0, n_prot, device))
                m = fmax_metrics_batched(fused, labels, true_per, has_label, thresholds_t, cc)

                precision_sum[c0:c1] += m["precision_sum"]
                precision_count[c0:c1] += m["precision_count"]
                recall_sum[c0:c1] += m["recall_sum"]
                default_tp[c0:c1] += m["default_tp"]
                default_pred_pos[c0:c1] += m["default_pred_pos"]
                total_pos[c0:c1] += m["total_pos"]
                tp_total[c0:c1] += m["tp_total"]
                del fused, m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                progress.update(c1 - c0)

            del P, labels, true_per, P_flat
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        progress.close()

    return {
        "precision_sum": precision_sum.cpu().numpy(),
        "precision_count": precision_count.cpu().numpy(),
        "recall_sum": recall_sum.cpu().numpy(),
        "recall_count": recall_count,
        "default_tp": default_tp.cpu().numpy(),
        "default_pred_pos": default_pred_pos.cpu().numpy(),
        "total_pos": total_pos.cpu().numpy(),
        "tp_total": tp_total.cpu().numpy(),
        "true_pos": true_pos_total,
    }


def candidate_fmax(metrics: dict) -> tuple[np.ndarray, np.ndarray]:
    ps = metrics["precision_sum"].astype(np.float64, copy=False)
    pc = metrics["precision_count"]
    rs = metrics["recall_sum"].astype(np.float64, copy=False)
    rc = metrics["recall_count"]
    prec = np.divide(ps, pc, out=np.zeros_like(ps), where=pc > 0)
    rec = rs / rc if rc > 0 else np.zeros_like(rs)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    best_idx = np.argmax(f1, axis=1)
    K = ps.shape[0]
    best_f = f1[np.arange(K), best_idx]
    best_thr = np.where(best_f > 0, THRESHOLDS[best_idx], 0.5)
    return best_f, best_thr


def candidate_aupr(metrics: dict) -> np.ndarray:
    tp = metrics["tp_total"].astype(np.float64, copy=False)
    pred = metrics["total_pos"].astype(np.float64, copy=False)
    true_pos = float(metrics["true_pos"])
    precision = np.divide(tp, pred, out=np.ones_like(tp), where=pred > 0)
    recall = tp / true_pos if true_pos > 0 else np.zeros_like(tp)
    recall_curve = np.concatenate([
        np.zeros((tp.shape[0], 1), dtype=np.float64),
        recall[:, ::-1],
        np.ones((tp.shape[0], 1), dtype=np.float64),
    ], axis=1)
    precision_curve = np.concatenate([
        np.ones((tp.shape[0], 1), dtype=np.float64),
        precision[:, ::-1],
        precision[:, :1],
    ], axis=1)
    return np.sum(np.diff(recall_curve, axis=1) * (precision_curve[:, :-1] + precision_curve[:, 1:]) / 2.0, axis=1)


def candidate_score(metrics: dict, fmax_alpha: float, aupr_beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fmax, _ = candidate_fmax(metrics)
    aupr = candidate_aupr(metrics)
    return fmax_alpha * fmax + aupr_beta * aupr, fmax, aupr


def constrain_blast_weight(W: np.ndarray, methods: Sequence[str], blast_max_weight: float) -> np.ndarray:
    if not 0.0 <= blast_max_weight <= 1.0:
        raise ValueError(f"blast_max_weight must be in [0, 1], got {blast_max_weight}")
    if "blast" not in methods:
        return W
    blast_idx = list(methods).index("blast")
    constrained = W[W[:, blast_idx] <= blast_max_weight + 1e-8]
    if constrained.size == 0:
        raise ValueError(
            f"No fusion candidates remain after applying blast_max_weight={blast_max_weight}"
        )
    return constrained


def best_candidate(metrics: dict, fmax_alpha: float = 1.0, aupr_beta: float = 0.0) -> tuple[int, dict]:
    score, best_f, aupr = candidate_score(metrics, fmax_alpha, aupr_beta)
    _, best_thr = candidate_fmax(metrics)
    winner = int(np.argmax(score))
    pred_pos = int(metrics["default_pred_pos"][winner])
    tp = int(metrics["default_tp"][winner])
    true_pos = int(metrics["true_pos"])
    return winner, {
        "score": float(score[winner]),
        "fmax": float(best_f[winner]),
        "aupr": float(aupr[winner]),
        "fmax_threshold": float(best_thr[winner]),
        "micro_precision": float(tp / pred_pos) if pred_pos > 0 else 0.0,
        "micro_recall": float(tp / true_pos) if true_pos > 0 else 0.0,
        "micro_f1": float(2 * tp / (pred_pos + true_pos)) if pred_pos + true_pos > 0 else 0.0,
    }


def search_weights(methods: Sequence[str], step: float, fold_data: List[dict], prop_t,
                   thresholds_t: torch.Tensor, device: torch.device,
                   desc: str, fmax_alpha: float, aupr_beta: float,
                   blast_max_weight: float, position: int = 0) -> tuple[np.ndarray, dict]:
    coarse_step = step * 2
    use_two_stage = True
    try:
        validate_step(coarse_step)
    except ValueError:
        use_two_stage = False

    if use_two_stage:
        W_coarse = constrain_blast_weight(simplex_grid(len(methods), coarse_step), methods, blast_max_weight)
        m_coarse = evaluate_candidates(W_coarse, fold_data, prop_t, thresholds_t,
                                        device, f"{desc} coarse", position)
        score_coarse, _, _ = candidate_score(m_coarse, fmax_alpha, aupr_beta)
        top_k = np.argsort(-score_coarse)[:2]
        W_fine = constrain_blast_weight(
            neighborhood_grid(W_coarse[top_k], step, radius=coarse_step),
            methods,
            blast_max_weight,
        )
        m_fine = evaluate_candidates(W_fine, fold_data, prop_t, thresholds_t,
                                      device, f"{desc} refine", position)
        winner, metrics = best_candidate(m_fine, fmax_alpha, aupr_beta)
        weights = W_fine[winner]
    else:
        W = constrain_blast_weight(simplex_grid(len(methods), step), methods, blast_max_weight)
        m = evaluate_candidates(W, fold_data, prop_t, thresholds_t, device, desc, position)
        winner, metrics = best_candidate(m, fmax_alpha, aupr_beta)
        weights = W[winner]
    return weights, metrics


def fuse_fold(weights: np.ndarray, fold: dict, prop_indices: list | None) -> np.ndarray:
    fused = np.tensordot(weights.astype(np.float32, copy=False), fold["P"], axes=(0, 0)).astype(np.float32, copy=False)
    if prop_indices is not None:
        fused = propagate_scores(fused, prop_indices).astype(np.float32, copy=False)
    return fused


def search_aspect(aspect: str, step: float, parents: dict | None, device: torch.device,
                  ic_path: Path, fmax_alpha: float, aupr_beta: float,
                  blast_max_weight: float, position: int = 0) -> dict:
    methods = discover_methods(aspect)
    classes = union_classes(methods, aspect)
    prop_indices = build_propagation_indices(classes, parents) if parents is not None else None
    prop_t = torch_prop_indices(prop_indices, device) if prop_indices is not None else None
    thr_dtype = torch.float32 if device.type == "mps" else torch.float64
    thresholds_t = torch.as_tensor(THRESHOLDS, device=device, dtype=thr_dtype)

    fold_data = [load_fold_cpu(methods, aspect, f, classes, prop_indices) for f in DEFAULT_FOLDS]
    n_prot = fold_data[0]["P"].shape[1]
    print(f"[{aspect}] device={device.type} methods={methods} n_classes={len(classes)} n_prot~{n_prot}")

    ic = load_information_content(ic_path, aspect, classes)
    nested_probs: List[np.ndarray] = []
    nested_labels: List[np.ndarray] = []
    heldout: Dict[int, dict] = {}
    for heldout_idx, heldout_fold in enumerate(DEFAULT_FOLDS):
        train_folds = [fd for i, fd in enumerate(fold_data) if i != heldout_idx]
        weights, train_metrics = search_weights(
            methods, step, train_folds, prop_t, thresholds_t, device,
            f"{aspect} nested fold{heldout_fold}", fmax_alpha, aupr_beta,
            blast_max_weight, position,
        )
        probs = fuse_fold(weights, fold_data[heldout_idx], prop_indices)
        labels = fold_data[heldout_idx]["labels"].astype(np.float32, copy=False)
        fold_metrics = compute_multilabel_metrics(labels, probs, 0.5, information_content=ic)
        nested_probs.append(probs)
        nested_labels.append(labels)
        heldout[heldout_fold] = {
            "weights": {meth: float(weights[i]) for i, meth in enumerate(methods)},
            "train_metrics": train_metrics,
            "heldout_metrics": fold_metrics,
        }
        print(
            f"{aspect} heldout_fold={heldout_fold} weights={heldout[heldout_fold]['weights']} "
            f"heldout_fmax={fold_metrics['fmax']:.4f} aupr={fold_metrics.get('aupr', 0.0):.4f} "
            f"smin={fold_metrics.get('smin', 0.0):.4f}"
        )

    nested_metrics = compute_multilabel_metrics(
        np.concatenate(nested_labels, axis=0),
        np.concatenate(nested_probs, axis=0),
        0.5,
        information_content=ic,
    )

    final_weights, final_metrics = search_weights(
        methods, step, fold_data, prop_t, thresholds_t, device, f"{aspect} final",
        fmax_alpha, aupr_beta, blast_max_weight, position,
    )

    return {
        "methods": methods,
        "weights": {meth: float(final_weights[i]) for i, meth in enumerate(methods)},
        "metrics": final_metrics,
        "nested_metrics": nested_metrics,
        "heldout": heldout,
    }


def main() -> None:
    args = parse_args()
    validate_step(args.step)
    parents = parse_go_obo(OBO_PATH) if args.propagate else None
    device = resolve_device(args.device)

    results: Dict[str, dict] = {}
    for i, aspect in enumerate(args.aspect):
        results[aspect] = search_aspect(
            aspect, args.step, parents, device, args.ic_pkl,
            args.fmax_alpha, args.aupr_beta, args.blast_max_weight, position=i,
        )

    rows: List[dict] = []
    summary: Dict[str, dict] = {}
    for aspect in args.aspect:
        best = results[aspect]
        methods = best["methods"]
        weights = best["weights"]
        metrics = best["metrics"]
        row = {"aspect": aspect, "thr": round(metrics["fmax_threshold"], 2)}
        for m in methods:
            row[f"w_{method_column(m)}"] = round(weights[m], 2)
        for m in methods:
            row[f"fold_{method_column(m)}"] = best_fold(m, aspect)
        nested = best["nested_metrics"]
        row["nested_fmax"] = round(nested["fmax"], 6)
        row["nested_aupr"] = round(nested.get("aupr", 0.0), 6)
        row["nested_smin"] = round(nested.get("smin", 0.0), 6)
        row["final_score"] = round(metrics.get("score", 0.0), 6)
        row["final_aupr"] = round(metrics.get("aupr", 0.0), 6)
        rows.append(row)
        summary[aspect] = {
            "weights": weights,
            "final_metrics": metrics,
            "nested_metrics": best["nested_metrics"],
            "heldout": best["heldout"],
            "folds": {m: row[f"fold_{method_column(m)}"] for m in methods},
            "search_objective": {
                "fmax_alpha": args.fmax_alpha,
                "aupr_beta": args.aupr_beta,
                "blast_max_weight": args.blast_max_weight,
            },
        }
        print(
            f"aspect={aspect} final_weights={weights} final_fmax={metrics['fmax']:.4f} "
            f"nested_fmax={nested['fmax']:.4f} nested_aupr={nested.get('aupr', 0.0):.4f} "
            f"nested_smin={nested.get('smin', 0.0):.4f}"
        )

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    with out_path.with_name(out_path.stem + "_summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2, sort_keys=True)
    print(f"Saved fusion weights to {out_path}")


if __name__ == "__main__":
    main()
