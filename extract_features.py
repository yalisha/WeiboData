"""Pipeline script to classify posts and derive daily features for time-series modeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

import classify_media


DEFAULT_IGNORE = {
    "modality": ["text_only", "image_only", "text_image"],
    "text_sentiment": ["positive", "negative", "neutral", "none"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("output/金价"), help="Directory containing per-day CSV exports.")
    parser.add_argument("--images-root", type=Path, default=Path("images/金价"), help="Root directory containing images.")
    parser.add_argument("--output", type=Path, default=Path("feature_exports/gold_features_daily.csv"), help="Where to write aggregated daily features.")
    parser.add_argument("--profile", default="mac-cpu", choices=["auto", "mac-cpu", "mac-mps", "gpu-server"], help="Hardware preset passed to classify_media.")
    parser.add_argument("--device", default="auto", help="Torch device override (optional).")
    parser.add_argument("--model", default=None, help="CLIP model architecture name.")
    parser.add_argument("--pretrained", default=None, help="Pretrained weights tag.")
    parser.add_argument("--batch-size", type=int, default=None, help="Image batch size for CLIP inference.")
    parser.add_argument("--text-weight", type=float, default=0.35, help="Weight for text embeddings when fusing with image embeddings.")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Low-confidence threshold for image classification.")
    parser.add_argument("--min-gap", type=float, default=0.03, help="Minimum difference between top-1 and top-2 similarities.")
    parser.add_argument("--image-proto-root", type=Path, default=Path("prototypes"), help="Prototype images directory for fine-grained classification.")
    parser.add_argument("--proto-text-weight", type=float, default=1.0, help="Weight multiplier for text prototypes.")
    parser.add_argument("--proto-image-weight", type=float, default=1.0, help="Weight multiplier for image prototypes.")
    parser.add_argument("--categories", type=Path, default=None, help="Optional JSON defining category prompts.")
    parser.add_argument("--start-date", help="Inclusive start date (YYYY-MM-DD). Processes all CSVs if omitted.")
    parser.add_argument("--end-date", help="Inclusive end date (YYYY-MM-DD). Processes all CSVs if omitted.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on posts for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; skip classification and feature aggregation.")
    parser.add_argument("--quality-sample", type=int, default=10, help="Number of rows to sample for manual QA (written to feature_exports/quality_samples.csv).")
    parser.add_argument("--quality-threshold", type=float, default=0.55, help="Confidence threshold for QA sampling.")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate prints.")
    return parser.parse_args()


def discover_dates(csv_root: Path, start: Optional[str], end: Optional[str]) -> Iterable[str]:
    dates = sorted(p.stem for p in csv_root.glob("*.csv"))
    if start is None and end is None:
        return dates
    filtered = []
    for date in dates:
        if start and date < start:
            continue
        if end and date > end:
            continue
        filtered.append(date)
    return filtered


def aggregate_daily_features(posts_df: pd.DataFrame) -> pd.DataFrame:
    if posts_df.empty:
        return pd.DataFrame()

    def load_json_column(series: pd.Series, keys: Optional[Iterable[str]] = None) -> pd.DataFrame:
        def parse_val(val: str) -> Dict[str, float]:
            if pd.isna(val) or not isinstance(val, str) or not val.strip():
                return {}
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            return {}
        json_dicts = series.apply(parse_val)
        expanded = pd.json_normalize(json_dicts)
        if keys:
            for key in keys:
                if key not in expanded:
                    expanded[key] = 0
        return expanded

    df = posts_df.copy()

    base_cols = [col for col in df.columns if col not in {"detail_tag", "detail_scores", "image_detail_counts", "image_category_counts", "image_detail_confidence", "image_detail_metadata"}]

    counts = df.groupby(["day"]).agg(
        posts=("post_id", "count"),
        posts_with_images=("has_image", lambda s: int(s.sum())),
        posts_with_text=("has_text", lambda s: int(s.sum())),
        images_total=("images_total", "sum"),
        images_classified_sum=("images_classified", "sum"),
        avg_image_confidence=("image_main_confidence", "mean"),
        low_conf_posts=("image_low_confidence_count", "sum"),
        ocr_count=("image_ocr_count", "sum"),
    )

    cat_counts = df.pivot_table(index="day", columns="image_main_category", values="post_id", aggfunc="count", fill_value=0)
    detail_counts = load_json_column(df["image_detail_counts"]).groupby(df["day"]).sum()
    detail_conf = load_json_column(df["image_detail_confidence"]).groupby(df["day"]).mean()

    sentiment_counts = df.pivot_table(index="day", columns="text_sentiment", values="post_id", aggfunc="count", fill_value=0)
    memo_cols = [col for col in sentiment_counts.columns if col not in DEFAULT_IGNORE["text_sentiment"]]
    if memo_cols:
        sentiment_counts = sentiment_counts.rename(columns={col: f"text_sentiment_{col}" for col in sentiment_counts.columns})

    modality_counts = df.pivot_table(index="day", columns="modality", values="post_id", aggfunc="count", fill_value=0)
    modality_counts = modality_counts.rename(columns={col: f"modality_{col}" for col in modality_counts.columns})

    topics_counts = df["text_topics"].str.get_dummies(sep=";").groupby(df["day"]).sum()
    topics_counts.columns = [f"topic_{col}" for col in topics_counts.columns]

    aggregated = counts
    for block in (cat_counts, detail_counts, detail_conf, sentiment_counts, modality_counts, topics_counts):
        if not block.empty:
            aggregated = aggregated.join(block, how="left")

    aggregated = aggregated.fillna(0)

    if aggregated.empty:
        return aggregated

    aggregated["images_per_post"] = aggregated.apply(lambda row: row["images_classified_sum"] / row["posts"] if row["posts"] else 0, axis=1)

    detail_cols = [col for col in aggregated.columns if col.startswith("technical_chart.") or col.startswith("news.") or col.startswith("gold_bullion.") or col.startswith("meme.")]
    if detail_cols:
        total_images = aggregated["images_classified_sum"].replace(0, pd.NA)
        for col in detail_cols:
            aggregated[f"ratio_{col}"] = aggregated[col] / total_images
        aggregated = aggregated.fillna(0)

    return aggregated.reset_index().sort_values("day")


def sample_for_quality(posts_df: pd.DataFrame, threshold: float, sample_size: int) -> pd.DataFrame:
    if posts_df.empty:
        return pd.DataFrame()
    priority = posts_df.copy()
    priority["_priority"] = 0
    if "image_main_confidence" in posts_df.columns:
        priority.loc[posts_df["image_main_confidence"] < threshold, "_priority"] += 1
    if "image_low_confidence_count" in posts_df.columns:
        priority.loc[posts_df["image_low_confidence_count"] > 0, "_priority"] += 1
    high_priority = priority[priority["_priority"] > 0]
    remaining = priority[priority["_priority"] == 0]
    frames = []
    if not high_priority.empty and sample_size > 0:
        frames.append(high_priority.sample(min(sample_size, len(high_priority)), random_state=42))
    taken = sum(len(frame) for frame in frames)
    additional = sample_size - taken
    if additional > 0 and not remaining.empty:
        frames.append(remaining.sample(min(additional, len(remaining)), random_state=42))
    if not frames:
        return pd.DataFrame()
    sample = pd.concat(frames).drop(columns=["_priority"], errors="ignore")
    return sample


def main() -> int:
    args = parse_args()
    dates = list(discover_dates(args.csv_root, args.start_date, args.end_date))
    if not dates:
        raise SystemExit("No CSV files found under the specified range.")

    all_features = []
    quality_samples: List[pd.DataFrame] = []

    for date in dates:
        classify_args = argparse.Namespace(
            images_root=args.images_root,
            csv_root=args.csv_root,
            output=args.output,
            aggregation="post",
            profile=args.profile,
            model=args.model,
            pretrained=args.pretrained,
            device=args.device,
            batch_size=args.batch_size,
            text_weight=args.text_weight,
            limit=args.limit,
            date=date,
            categories=args.categories,
            min_confidence=args.min_confidence,
            min_gap=args.min_gap,
            image_proto_root=args.image_proto_root,
            proto_text_weight=args.proto_text_weight,
            proto_image_weight=args.proto_image_weight,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )

        stats, posts, datapoints, posts_df, images_df = classify_media.run_classification(classify_args)

        if args.dry_run:
            if not args.quiet:
                print(f"Dry-run: {date} posts={stats['posts']} images={stats['images']}")
            continue

        if posts_df is None or posts_df.empty:
            continue

        features = aggregate_daily_features(posts_df)
        if not features.empty:
            all_features.append(features)

        sample_df = sample_for_quality(posts_df, args.quality_threshold, args.quality_sample)
        if not sample_df.empty:
            sample_df.insert(0, "sample_date", date)
            quality_samples.append(sample_df)

    if args.dry_run:
        return 0

    if not all_features:
        raise SystemExit("No features were generated. Check inputs or filters.")

    feature_table = pd.concat(all_features, ignore_index=True)
    feature_table = feature_table.sort_values("day")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.output, index=False)
    if not args.quiet:
        print(f"Wrote features to {args.output}")

    if quality_samples:
        qa_path = args.output.parent / "quality_samples.csv"
        pd.concat(quality_samples, ignore_index=True).to_csv(qa_path, index=False)
        if not args.quiet:
            print(f"Wrote quality samples to {qa_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
