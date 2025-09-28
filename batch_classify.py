"""Batch classification driver for social media multimodal data.

This script orchestrates `classify_media.py` to generate both post-level and
image-level outputs across multiple CSV files. It keeps the per-image details
for debugging while exporting a clean per-post summary for downstream feature
engineering or model training.

Example usage:

    python batch_classify.py \
        --csv-root output/金价 \
        --images-root images/金价 \
        --output-root batch_results \
        --date 2022-01-01 \
        --profile mac-cpu \
        --image-proto-root prototypes \
        --keep-image-level
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

import classify_media


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, required=True, help="Directory containing daily CSV files.")
    parser.add_argument("--images-root", type=Path, default=None, help="Root directory for downloaded images (optional).")
    parser.add_argument("--output-root", type=Path, default=Path("batch_results"), help="Directory to store aggregated outputs.")
    parser.add_argument("--date", action="append", help="Specific YYYY-MM-DD date(s) to process; repeatable.")
    parser.add_argument("--start-date", help="Inclusive start date (YYYY-MM-DD). Process all available dates if omitted.")
    parser.add_argument("--end-date", help="Inclusive end date (YYYY-MM-DD). Process all available dates if omitted.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of posts/images per run (debug).")
    parser.add_argument("--profile", default="auto", choices=["auto", "mac-cpu", "mac-mps", "gpu-server"], help="Hardware preset for CLIP inference.")
    parser.add_argument("--device", default="auto", help="Torch device string to override `--profile`.")
    parser.add_argument("--model", default=None, help="CLIP backbone name (overrides profile).")
    parser.add_argument("--pretrained", default=None, help="Pretrained weights tag for CLIP model.")
    parser.add_argument("--batch-size", type=int, default=None, help="Images per batch during CLIP inference.")
    parser.add_argument("--text-weight", type=float, default=0.35, help="Weight of text embedding when fusing with image features.")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Low-confidence threshold for image classification.")
    parser.add_argument("--min-gap", type=float, default=0.03, help="Minimum margin between top-1/top-2 similarity scores.")
    parser.add_argument("--image-proto-root", type=Path, default=None, help="Directory containing category-specific prototype images.")
    parser.add_argument("--proto-text-weight", type=float, default=1.0, help="Weight multiplier for text-based prototypes.")
    parser.add_argument("--proto-image-weight", type=float, default=1.0, help="Weight multiplier for image-based prototypes.")
    parser.add_argument("--categories", type=Path, default=None, help="Optional JSON file overriding category prompts.")
    parser.add_argument("--keep-image-level", action="store_true", help="Persist per-image classification CSV alongside post-level output.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging in the underlying classifier.")
    parser.add_argument("--dry-run", action="store_true", help="Preview which posts would be processed without running CLIP.")
    return parser.parse_args()


def discover_dates(
    csv_root: Path,
    selected: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[str]:
    if selected:
        candidates = sorted(set(selected))
    else:
        candidates = sorted(p.stem for p in csv_root.glob("*.csv"))

    if start is None and end is None:
        return candidates

    filtered: List[str] = []
    for date in candidates:
        if start and date < start:
            continue
        if end and date > end:
            continue
        filtered.append(date)
    return filtered


def build_args(base: argparse.Namespace, date: str, aggregation: str, output_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        images_root=base.images_root,
        csv_root=base.csv_root,
        output=output_path,
        aggregation=aggregation,
        profile=base.profile,
        model=base.model,
        pretrained=base.pretrained,
        device=base.device,
        batch_size=base.batch_size,
        text_weight=base.text_weight,
        limit=base.limit,
        date=date,
        categories=base.categories,
        min_confidence=base.min_confidence,
        min_gap=base.min_gap,
        image_proto_root=base.image_proto_root,
        proto_text_weight=base.proto_text_weight,
        proto_image_weight=base.proto_image_weight,
        dry_run=base.dry_run,
        verbose=base.verbose,
    )


def write_dataframe(df: Optional[pd.DataFrame], path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    dates = discover_dates(
        args.csv_root,
        selected=args.date,
        start=args.start_date,
        end=args.end_date,
    )
    if not dates:
        raise SystemExit("No CSV files discovered to process.")

    summary_records = []

    for date in dates:
        post_args = build_args(args, date=date, aggregation="post", output_path=output_root / "posts" / f"{date}.csv")

        stats, posts, datapoints, posts_df, images_df = classify_media.run_classification(post_args)
        summary_records.append(
            {
                "date": date,
                "csv_files": stats["csv_files"],
                "posts": stats["posts"],
                "posts_with_images": stats["posts_with_images"],
                "images": stats["images"],
                "dry_run": args.dry_run,
            }
        )

        if args.dry_run:
            continue

        write_dataframe(posts_df, post_args.output)

        if args.keep_image_level:
            image_path = output_root / "images" / f"{date}.csv"
            write_dataframe(images_df, image_path)

    summary_df = pd.DataFrame.from_records(summary_records)
    summary_path = output_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")

    if args.dry_run:
        print("Dry run complete; no classification outputs were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
