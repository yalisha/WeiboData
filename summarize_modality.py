"""Summarise modality coverage for collected social posts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import pandas as pd


@dataclass
class ModalityTally:
    total: int = 0
    text_only: int = 0
    image_only: int = 0
    text_image: int = 0
    empty: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "text_only": self.text_only,
            "image_only": self.image_only,
            "text_image": self.text_image,
            "empty": self.empty,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_root",
        type=Path,
        help="Directory containing daily CSV exports (e.g. output/金价).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modality_stats.csv"),
        help="Where to write the summary CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of CSV files processed (for quick tests).",
    )
    return parser.parse_args()


def iter_csv_files(root: Path, limit: Optional[int]) -> Iterable[Path]:
    files = sorted(p for p in root.glob("*.csv") if p.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def _normalize(value: Union[str, float, None]) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return ""
    return str(value)


def detect_modality(text: Optional[str], image_paths: Optional[str]) -> str:
    has_text = bool(_normalize(text).strip())
    has_image = bool(_normalize(image_paths).strip())
    if has_text and has_image:
        return "text_image"
    if has_text:
        return "text_only"
    if has_image:
        return "image_only"
    return "empty"


def tally_dataframe(df: pd.DataFrame) -> ModalityTally:
    tally = ModalityTally()
    for _, row in df.iterrows():
        modality = detect_modality(row.get("text"), row.get("image_paths"))
        tally.total += 1
        if modality == "text_only":
            tally.text_only += 1
        elif modality == "image_only":
            tally.image_only += 1
        elif modality == "text_image":
            tally.text_image += 1
        else:
            tally.empty += 1
    return tally


def main() -> int:
    args = parse_args()
    rows = []
    overall = ModalityTally()

    for csv_file in iter_csv_files(args.csv_root, args.limit):
        df = pd.read_csv(csv_file)
        tally = tally_dataframe(df)
        for key, value in tally.as_dict().items():
            setattr(overall, key, getattr(overall, key) + value)
        day = df.get("day")
        day_value = day.iloc[0] if isinstance(day, pd.Series) and not day.empty else csv_file.stem
        rows.append({
            "csv_path": str(csv_file),
            "day": day_value,
            **tally.as_dict(),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output, index=False)

    overall_df = pd.DataFrame([
        {"csv_path": "TOTAL", "day": "TOTAL", **overall.as_dict()},
    ])
    combined = pd.concat([summary_df, overall_df], ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {args.output} (rows: {len(summary_df)})")
    print(overall_df.iloc[0].to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
