"""Quality inspection utilities for aggregated classification outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", type=Path, required=True, help="Post-level classification CSV produced by `classify_media.py`." )
    parser.add_argument("--images", type=Path, default=None, help="Optional image-level classification CSV for deeper inspection.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional CSV with manual labels (columns: post_id, expected_detail_tag[, expected_sentiment]).")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of rows to sample for manual review.")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Flag posts whose image_main_confidence falls below this threshold.")
    parser.add_argument("--output", type=Path, default=Path("quality_report.json"), help="Path to save aggregated statistics (JSON).")
    parser.add_argument("--sample-output", type=Path, default=Path("quality_samples.csv"), help="CSV to store sampled rows for manual review.")
    return parser.parse_args()


def load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def summarize_posts(df: pd.DataFrame, threshold: float) -> Dict[str, any]:
    summary: Dict[str, any] = {}
    summary["total_posts"] = int(len(df))
    summary["modality_counts"] = df["modality"].value_counts().to_dict()
    if "text_sentiment" in df.columns:
        summary["text_sentiment_counts"] = df["text_sentiment"].value_counts().to_dict()
    if "image_main_category" in df.columns:
        summary["image_main_category_counts"] = df["image_main_category"].value_counts(dropna=True).to_dict()
    if "image_detail_tag" in df.columns:
        summary["image_detail_tag_counts"] = df["image_detail_tag"].value_counts(dropna=True).to_dict()

    weak_conf_mask = df.get("image_main_confidence", pd.Series(dtype=float)) < threshold
    if weak_conf_mask.any():
        summary["low_confidence_posts"] = int(weak_conf_mask.sum())
    else:
        summary["low_confidence_posts"] = 0

    detail_counts = df.get("image_detail_counts")
    if detail_counts is not None:
        try:
            expanded = detail_counts.dropna().apply(lambda x: json.loads(x) if isinstance(x, str) else {})
            aggregate: Dict[str, int] = {}
            for item in expanded:
                for key, value in item.items():
                    aggregate[key] = aggregate.get(key, 0) + int(value)
            summary["detail_tag_totals"] = aggregate
        except json.JSONDecodeError:
            pass

    return summary


def compute_classification_metrics(pred: pd.Series, truth: pd.Series) -> Dict[str, float]:
    assert len(pred) == len(truth)
    mask = truth.notna() & pred.notna()
    pred = pred[mask]
    truth = truth[mask]
    if pred.empty:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}

    labels = sorted(set(truth.unique()) | set(pred.unique()))
    confusion = {label: {l: 0 for l in labels} for label in labels}
    for p, t in zip(pred, truth):
        confusion[t][p] += 1

    accuracy = sum(confusion[label].get(label, 0) for label in labels) / len(pred)

    f1_scores: List[float] = []
    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in labels if other != label)
        fn = sum(confusion[label].get(other, 0) for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else float("nan")
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4)}


def evaluate_against_ground_truth(posts_df: pd.DataFrame, truth_path: Path) -> Dict[str, Dict[str, float]]:
    truth_df = pd.read_csv(truth_path)
    required_cols = {"post_id", "expected_detail_tag"}
    if not required_cols.issubset(truth_df.columns):
        raise ValueError(f"Ground truth CSV must contain columns: {required_cols}")

    merged = posts_df.merge(truth_df, on="post_id", how="inner", suffixes=("_pred", "_true"))
    metrics: Dict[str, Dict[str, float]] = {}
    metrics["detail_tag"] = compute_classification_metrics(merged["image_detail_tag"], merged["expected_detail_tag"])

    if "expected_sentiment" in merged.columns and "text_sentiment" in merged.columns:
        metrics["text_sentiment"] = compute_classification_metrics(
            merged["text_sentiment"], merged["expected_sentiment"]
        )
    return metrics


def sample_for_review(df: pd.DataFrame, sample_size: int, threshold: float) -> pd.DataFrame:
    priority = df.copy()
    priority["_priority"] = 0
    if "image_main_confidence" in df.columns:
        priority.loc[df["image_main_confidence"] < threshold, "_priority"] += 1
    if "image_low_confidence_count" in df.columns:
        priority.loc[df["image_low_confidence_count"] > 0, "_priority"] += 1

    high_priority = priority[priority["_priority"] > 0]
    remaining = priority[priority["_priority"] == 0]

    samples: List[pd.DataFrame] = []
    if not high_priority.empty:
        samples.append(high_priority.sample(min(sample_size, len(high_priority)), random_state=42))
    if len(samples) < 1 and not remaining.empty:
        samples.append(remaining.sample(min(sample_size, len(remaining)), random_state=42))
    elif not remaining.empty and sample_size > len(samples[0]):
        additional = sample_size - len(samples[0])
        samples.append(remaining.sample(min(additional, len(remaining)), random_state=42))

    if not samples:
        return pd.DataFrame()
    combined = pd.concat(samples).drop(columns=["_priority"], errors="ignore")
    return combined


def main() -> int:
    args = parse_args()
    posts_df = load_csv(args.posts)
    if posts_df is None:
        raise SystemExit("Post-level CSV is required for quality checks.")
    images_df = load_csv(args.images)

    summary = summarize_posts(posts_df, args.confidence_threshold)

    if args.ground_truth:
        summary["metrics"] = evaluate_against_ground_truth(posts_df, args.ground_truth)

    sample_df = sample_for_review(posts_df, args.sample_size, args.confidence_threshold)
    if not sample_df.empty:
        args.sample_output.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(args.sample_output, index=False)
        summary["sample_output"] = str(args.sample_output)

    if images_df is not None and not images_df.empty:
        summary["image_records"] = int(len(images_df))
        summary["image_categories"] = images_df["category"].value_counts().to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
