"""Gemini/OpenAI-based multimodal classification and daily feature aggregation.

This script mirrors the structure of ``classify_media.py``/``extract_features.py``
but swaps out the CLIP pipeline for Google Gemini or any OpenAI-compatible
multimodal API. It reads raw per-day post dumps, requests the remote model to
label sentiment/topics/images, and optionally aggregates the results with
``aggregate_daily_features``.

Usage example::

    python gemini_feature_pipeline.py \
        --csv-root output/金价 \
        --images-root images/金价 \
        --classified-output feature_exports/classified_gemini.csv \
        --output feature_exports/gold_features_daily_gemini.csv \
        --enable-interactions \
        --mock

Set ``GEMINI_API_KEY`` (Google provider) or ``OPENAI_API_KEY``/``--api-key`` for
OpenAI-compatible endpoints (e.g. http://104.225.150.15:7860/v1) before running
in live mode.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

import requests

try:  # Optional dependency; users may prefer mock mode during development.
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - allow running without the SDK.
    genai = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress bar is optional.
    tqdm = None  # type: ignore

from extract_features import aggregate_daily_features


LOGGER = logging.getLogger("gemini_pipeline")


MAIN_CATEGORIES: Sequence[str] = (
    "technical_chart",
    "news_screenshot",
    "gold_bullion",
    "meme",
    "irrelevant",
)

DETAIL_TAGS: Sequence[str] = (
    "technical_chart.up",
    "technical_chart.down",
    "technical_chart.sideways",
    "technical_chart.uncertain",
    "gold_bullion.bar",
    "gold_bullion.coin",
    "gold_bullion.jewelry",
    "gold_bullion.packaging",
    "news.positive",
    "news.neutral",
    "news.negative",
    "meme.positive",
    "meme.neutral",
    "meme.negative",
    "noise",
)

TEXT_SENTIMENTS: Sequence[str] = ("positive", "negative", "neutral", "none")

TOPIC_NAMES: Sequence[str] = ("macro", "technical", "jewelry", "meme_text", "risk")


POSITIVE_TERMS = {
    "上涨",
    "大涨",
    "利好",
    "赚钱",
    "盈利",
    "走强",
    "看涨",
    "高位",
    "突破",
    "乐观",
    "飙升",
    "涨停",
    "走高",
    "飙涨",
    "复苏",
    "增持",
    "买入",
    "强势",
}

NEGATIVE_TERMS = {
    "下跌",
    "大跌",
    "利空",
    "亏损",
    "风险",
    "抛售",
    "看跌",
    "暴跌",
    "回调",
    "崩盘",
    "暴雷",
    "踩雷",
    "爆仓",
    "承压",
    "缩水",
    "疲软",
}

TOPIC_KEYWORDS: Dict[str, Sequence[str]] = {
    "macro": ("美联储", "央行", "利率", "通胀", "加息", "宏观", "货币政策", "GDP"),
    "technical": ("支撑", "阻力", "K线", "均线", "技术面", "形态", "指标", "波段"),
    "jewelry": ("首饰", "珠宝", "金店", "克价", "黄金首饰", "首饰店"),
    "meme_text": ("哈哈", "笑死", "梗", "表情包", "段子", "吐槽"),
    "risk": ("风险", "暴跌", "爆仓", "踩雷", "危机", "衰退"),
}


@dataclass
class PostRecord:
    post_id: str
    day: str
    text: str
    image_paths: List[Path]
    csv_path: Path
    row_index: int

    @property
    def has_text(self) -> bool:
        return bool(self.text and str(self.text).strip())

    @property
    def has_image(self) -> bool:
        return bool(self.image_paths)

    @property
    def modality(self) -> str:
        if self.has_text and self.has_image:
            return "text_image"
        if self.has_image:
            return "image_only"
        if self.has_text:
            return "text_only"
        return "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("output/金价"), help="Directory containing per-day raw CSV exports.")
    parser.add_argument("--images-root", type=Path, default=Path("images/金价"), help="Directory containing downloaded images.")
    parser.add_argument("--output", type=Path, default=Path("feature_exports/gold_features_daily_gemini.csv"), help="Where to write aggregated daily features.")
    parser.add_argument("--classified-output", type=Path, default=Path("feature_exports/classified_posts_gemini.csv"), help="Optional CSV to persist per-post Gemini classifications.")
    parser.add_argument("--provider", choices=["google", "openai"], default="google", help="Backend provider: Google Gemini SDK or OpenAI-compatible endpoint.")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash", help="Gemini model name (e.g. gemini-1.5-pro, gemini-1.5-flash).")
    parser.add_argument("--openai-base-url", default="http://104.225.150.15:7860/v1", help="Base URL for OpenAI-compatible endpoints (set when --provider=openai).")
    parser.add_argument("--openai-model", default="gemini-2.5-flash", help="Model name used for OpenAI-compatible provider.")
    parser.add_argument("--api-key", default=None, help="Gemini API key (overrides GEMINI_API_KEY environment variable).")
    parser.add_argument("--start-date", help="Inclusive start date (YYYY-MM-DD). Processes all CSVs if omitted.")
    parser.add_argument("--end-date", help="Inclusive end date (YYYY-MM-DD). Processes all CSVs if omitted.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on posts per day for debugging.")
    parser.add_argument("--max-images", type=int, default=3, help="Max number of images per post to send to Gemini (cost control).")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.45, help="Images below this confidence count as low-confidence.")
    parser.add_argument("--request-pause", type=float, default=0.0, help="Seconds to sleep between Gemini calls (rate limit safety).")
    parser.add_argument("--cache-dir", type=Path, default=Path("feature_exports/gemini_cache"), help="Cache directory for Gemini JSON responses.")
    parser.add_argument("--mock", action="store_true", help="Skip real API calls and generate deterministic mock classifications.")
    parser.add_argument("--enable-interactions", action="store_true", help="Pass through to aggregate_daily_features to add interaction columns.")
    parser.add_argument("--dry-run", action="store_true", help="Only produce per-post classifications; skip aggregation and file writes.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output.")
    return parser.parse_args()


def discover_dates(csv_root: Path, start: Optional[str], end: Optional[str]) -> Iterable[str]:
    dates = sorted(p.stem for p in csv_root.glob("*.csv"))
    if start is None and end is None:
        return dates
    filtered: List[str] = []
    for date in dates:
        if start and date < start:
            continue
        if end and date > end:
            continue
        filtered.append(date)
    return filtered


class GeminiClassifier:
    """Wrapper around the Gemini API with caching and mock support."""

    def __init__(
        self,
        provider: str,
        gemini_model: str,
        openai_model: str,
        openai_base_url: str,
        api_key: Optional[str],
        cache_dir: Path,
        max_images: int,
        mock: bool,
        request_pause: float,
        low_confidence_threshold: float,
    ) -> None:
        self.provider = provider
        self.model_name = gemini_model
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url.rstrip("/")
        self.mock = mock
        self.cache_dir = cache_dir
        self.max_images = max_images
        self.request_pause = request_pause
        self.low_confidence_threshold = low_confidence_threshold

        self._api_key = api_key

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.mock:
            self.model = None
            return

        if self.provider == "google":
            if genai is None:  # pragma: no cover - require user installation
                raise SystemExit(
                    "google-generativeai is not installed. Install with `pip install google-generativeai` or use --mock."
                )

            final_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not final_key:
                raise SystemExit("Gemini API key missing. Set GEMINI_API_KEY or provide --api-key.")

            genai.configure(api_key=final_key)
            self.model = genai.GenerativeModel(gemini_model)
        else:
            final_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not final_key:
                raise SystemExit("OpenAI-compatible provider requires an API key. Pass --api-key or set OPENAI_API_KEY.")
            self.model = None
            self._api_key = final_key

    def classify(self, post: PostRecord) -> Dict[str, Any]:
        cache_path = self.cache_dir / f"{post.post_id}_{post.row_index}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                LOGGER.warning("Cache file %s is corrupt; regenerating.", cache_path)

        if self.mock:
            result = self._mock_response(post)
        else:
            if self.provider == "google":
                result = self._call_google(post)
            else:
                result = self._call_openai(post)

        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def _call_google(self, post: PostRecord) -> Dict[str, Any]:  # pragma: no cover - network call
        assert self.model is not None

        prompt = self._build_prompt(post)

        parts: List[Any] = [prompt]
        for image_path in post.image_paths[: self.max_images]:
            try:
                parts.append(self._image_part(image_path))
            except FileNotFoundError:
                LOGGER.warning("Image missing: %s", image_path)

        generation_config = {
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "top_p": 0.95,
            "top_k": 40,
        }

        response = self.model.generate_content(parts, generation_config=generation_config)
        if hasattr(response, "text"):
            raw_text = response.text
        elif hasattr(response, "result") and response.result:  # pylint:disable=access-member-before-definition
            raw_text = response.result
        else:
            raw_text = ""

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse Gemini response for %s: %s", post.post_id, exc)
            data = self._keyword_baseline(post)

        if self.request_pause:
            time.sleep(self.request_pause)

        return data

    def _call_openai(self, post: PostRecord) -> Dict[str, Any]:  # pragma: no cover - network call
        prompt = self._build_prompt(post)

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt}
        ]

        for image_path in post.image_paths[: self.max_images]:
            try:
                image_bytes = image_path.read_bytes()
            except FileNotFoundError:
                LOGGER.warning("Image missing: %s", image_path)
                continue
            mime = "image/jpeg"
            suffix = image_path.suffix.lower()
            if suffix == ".png":
                mime = "image/png"
            elif suffix == ".webp":
                mime = "image/webp"
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                }
            )

        payload = {
            "model": self.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a multimodal assistant that returns strict JSON responses."}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": 0.2,
        }

        url = f"{self.openai_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as exc:
            LOGGER.error("OpenAI-compatible request failed for %s: %s", post.post_id, exc)
            return self._keyword_baseline(post)

        if response.status_code != 200:
            LOGGER.error("OpenAI-compatible endpoint error %s: %s", response.status_code, response.text[:200])
            return self._keyword_baseline(post)

        try:
            data = response.json()
        except ValueError:
            LOGGER.error("Invalid JSON response from OpenAI-compatible endpoint for %s", post.post_id)
            return self._keyword_baseline(post)

        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    content_parts.append(part.get("text", ""))
            content = "\n".join(content_parts)

        if not isinstance(content, str):
            LOGGER.error("Unexpected message content type from OpenAI-compatible endpoint for %s", post.post_id)
            return self._keyword_baseline(post)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            LOGGER.error("Assistant response not valid JSON for %s: %s", post.post_id, content[:200])
            parsed = self._keyword_baseline(post)

        if self.request_pause:
            time.sleep(self.request_pause)

        return parsed

    @staticmethod
    def _image_part(image_path: Path) -> Any:  # pragma: no cover - relies on SDK type definitions
        mime = "image/jpeg"
        if image_path.suffix.lower() == ".png":
            mime = "image/png"
        elif image_path.suffix.lower() in {".webp", ".gif"}:
            mime = f"image/{image_path.suffix.lower().lstrip('.')}"
        image_bytes = image_path.read_bytes()
        if genai is not None:
            return genai.types.Part.from_data(mime_type=mime, data=image_bytes)
        # Fallback: encode in base64 for compatibility with REST-style payloads.
        return {
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(image_bytes).decode("utf-8"),
            }
        }

    @staticmethod
    def _build_prompt(post: PostRecord) -> str:
        truncated_text = (post.text or "").strip()
        if len(truncated_text) > 800:
            truncated_text = truncated_text[:800] + " …"

        template = {
            "task": "classify_gold_social_post",
            "instructions": textwrap.dedent(
                """
                You are a multimodal financial assistant. Review the provided text and images
                (image count may be zero) and produce standardized annotations:
                  1. Sentiment label must be one of: positive, negative, neutral, none.
                  2. Topics may be chosen from: {topics}. Return an empty list if none apply.
                  3. For each image, provide a main_category from: {categories}. Optionally supply
                     a detail_tag such as technical_chart.up or gold_bullion.bar when confident.
                  4. Respond with JSON only, using this structure exactly:
                     - "sentiment": {{"label": <string>, "confidence": <0-1 float>}}
                     - "topics": [{{"name": <string>, "confidence": <0-1 float>}}, ...]
                     - "images": [{{"index": <int>, "main_category": {{"label": <string>, "confidence": <0-1 float>}},
                       "detail_tag": <string or null>, "detail_confidence": <0-1 float or null>}}, ...]
                Return JSON only with double-quoted keys and values.
                """
            ).format(
                topics=", ".join(TOPIC_NAMES),
                categories=", ".join(MAIN_CATEGORIES),
            ),
            "post": {
                "post_id": post.post_id,
                "day": post.day,
                "text": truncated_text,
                "has_text": post.has_text,
                "image_count": len(post.image_paths),
            },
        }
        return json.dumps(template, ensure_ascii=False)

    def _mock_response(self, post: PostRecord) -> Dict[str, Any]:
        random.seed(hash((post.post_id, post.row_index)))

        sentiment_label = random.choice(TEXT_SENTIMENTS)
        topics = []
        for topic in TOPIC_NAMES:
            if random.random() < 0.2:
                topics.append({"name": topic, "confidence": round(random.uniform(0.4, 0.9), 2)})

        images: List[Dict[str, Any]] = []
        for idx, _ in enumerate(post.image_paths[: self.max_images]):
            label = random.choice(MAIN_CATEGORIES)
            detail = None
            detail_conf = None
            if label.startswith("technical_chart"):
                detail = random.choice([
                    "technical_chart.up",
                    "technical_chart.down",
                    "technical_chart.sideways",
                ])
            elif label == "gold_bullion":
                detail = random.choice([
                    "gold_bullion.bar",
                    "gold_bullion.coin",
                    "gold_bullion.jewelry",
                    "gold_bullion.packaging",
                ])
            elif label == "meme":
                detail = random.choice(["meme.positive", "meme.neutral", "meme.negative"])
            elif label == "news_screenshot":
                detail = random.choice(["news.positive", "news.neutral", "news.negative"])
            if detail:
                detail_conf = round(random.uniform(0.4, 0.9), 2)
            images.append(
                {
                    "index": idx,
                    "main_category": {
                        "label": label,
                        "confidence": round(random.uniform(0.35, 0.95), 3),
                    },
                    "detail_tag": detail,
                    "detail_confidence": detail_conf,
                }
            )

        return {
            "sentiment": {"label": sentiment_label, "confidence": round(random.uniform(0.4, 0.9), 3)},
            "topics": topics,
            "images": images,
        }

    def _keyword_baseline(self, post: PostRecord) -> Dict[str, Any]:
        text = post.text or ""
        sentiment = "neutral"
        for term in POSITIVE_TERMS:
            if term in text:
                sentiment = "positive"
                break
        else:
            for term in NEGATIVE_TERMS:
                if term in text:
                    sentiment = "negative"
                    break

        topics = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(word in text for word in keywords):
                topics.append({"name": topic, "confidence": 0.6})

        return {
            "sentiment": {"label": sentiment, "confidence": 0.55},
            "topics": topics,
            "images": [
                {
                    "index": idx,
                    "main_category": {
                        "label": "irrelevant",
                        "confidence": 0.5,
                    },
                    "detail_tag": "noise",
                    "detail_confidence": 0.5,
                }
                for idx, _ in enumerate(post.image_paths[: self.max_images])
            ],
        }


def read_daily_posts(csv_path: Path, images_root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["image_paths"] = df.get("image_paths", "").fillna("")
    df["image_paths"] = df["image_paths"].apply(lambda p: _parse_image_paths(p, images_root))
    df["text"] = df.get("text", "").fillna("")
    return df


def _parse_image_paths(raw: Any, images_root: Path) -> List[Path]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [segment.strip() for segment in raw.split(";") if segment.strip()]
    return [Path(part) if part.startswith("/") else images_root / part for part in parts]


def classify_posts(df: pd.DataFrame, csv_path: Path, classifier: GeminiClassifier) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    iterator = df.iterrows()
    if tqdm is not None and len(df) > 10:
        iterator = tqdm(iterator, total=len(df), desc=f"Gemini {csv_path.stem}")  # type: ignore

    for idx, row in iterator:  # type: ignore
        post = PostRecord(
            post_id=str(row.get("post_id", f"{csv_path.stem}_{idx}")),
            day=str(row.get("day", csv_path.stem)),
            text=str(row.get("text", "")),
            image_paths=list(row.get("image_paths", [])),
            csv_path=csv_path,
            row_index=int(idx),
        )

        result = classifier.classify(post)
        records.append(
            build_row(
                post,
                result,
                classifier.max_images,
                classifier.request_pause,
                classifier.mock,
                classifier.low_confidence_threshold,
            )
        )

    return pd.DataFrame.from_records(records)


def build_row(
    post: PostRecord,
    result: Dict[str, Any],
    max_images: int,
    request_pause: float,
    mock_mode: bool,
    low_conf_threshold: float,
) -> Dict[str, Any]:
    sentiment = result.get("sentiment", {})
    sentiment_label = sentiment.get("label") or "none"
    if sentiment_label not in TEXT_SENTIMENTS:
        sentiment_label = "none"
    sentiment_conf = _safe_float(sentiment.get("confidence"), default=0.0)

    topic_entries = []
    for topic in result.get("topics", []) or []:
        name = str(topic.get("name", "")).strip()
        if name in TOPIC_NAMES:
            topic_entries.append(name)
    text_topics = ";".join(sorted(set(topic_entries)))

    images_data = result.get("images", []) or []
    image_rows: List[Dict[str, Any]] = []
    for item in images_data:
        index = int(item.get("index", len(image_rows)))
        main_cat = item.get("main_category", {}) or {}
        label = str(main_cat.get("label", "")).strip() or "irrelevant"
        if label not in MAIN_CATEGORIES:
            label = "irrelevant"
        conf = _safe_float(main_cat.get("confidence"), default=0.0)
        detail_tag = item.get("detail_tag")
        if detail_tag is None:
            detail_tag = ""
        detail_tag = str(detail_tag).strip()
        if detail_tag and detail_tag not in DETAIL_TAGS:
            detail_tag = ""
        detail_conf = _safe_float(item.get("detail_confidence"), default=0.0)

        image_rows.append(
            {
                "index": index,
                "main_category": label,
                "main_confidence": conf,
                "detail_tag": detail_tag,
                "detail_confidence": detail_conf,
            }
        )

    # Ensure consistency even when Gemini returns fewer entries than images supplied.
    if not image_rows and post.has_image:
        for idx in range(min(len(post.image_paths), max_images)):
            image_rows.append(
                {
                    "index": idx,
                    "main_category": "irrelevant",
                    "main_confidence": 0.0,
                    "detail_tag": "",
                    "detail_confidence": 0.0,
                }
            )

    category_counts: Dict[str, int] = {}
    category_conf: Dict[str, float] = {}
    detail_counts: Dict[str, int] = {}
    detail_conf: Dict[str, float] = {}
    detail_meta: List[Dict[str, Any]] = []

    for img in image_rows:
        cat = img["main_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        category_conf[cat] = category_conf.get(cat, 0.0) + img["main_confidence"]

        if img["detail_tag"]:
            detail = img["detail_tag"]
            detail_counts[detail] = detail_counts.get(detail, 0) + 1
            detail_conf[detail] = detail_conf.get(detail, 0.0) + img["detail_confidence"]

        detail_meta.append(
            {
                "index": img["index"],
                "main_category": img["main_category"],
                "main_confidence": img["main_confidence"],
                "detail_tag": img["detail_tag"],
                "detail_confidence": img["detail_confidence"],
            }
        )

    for cat in list(category_conf):
        category_conf[cat] = category_conf[cat] / max(category_counts[cat], 1)
    for detail in list(detail_conf):
        detail_conf[detail] = detail_conf[detail] / max(detail_counts[detail], 1)

    dominant_category = max(category_counts.items(), key=lambda kv: kv[1], default=("irrelevant", 0))[0]
    dominant_conf = category_conf.get(dominant_category, 0.0)

    dominant_detail = max(detail_counts.items(), key=lambda kv: kv[1], default=("", 0))[0]
    image_low_conf = sum(1 for img in image_rows if img["main_confidence"] < low_conf_threshold)

    return {
        "csv_path": str(post.csv_path),
        "post_id": post.post_id,
        "day": post.day,
        "row_index": post.row_index,
        "modality": post.modality,
        "has_text": post.has_text,
        "has_image": post.has_image,
        "images_total": len(post.image_paths),
        "image_main_category": dominant_category,
        "image_detail_tag": dominant_detail,
        "image_main_confidence": round(dominant_conf, 4),
        "images_classified": len(image_rows),
        "image_category_counts": json.dumps(category_counts, ensure_ascii=False),
        "image_category_confidence": json.dumps(category_conf, ensure_ascii=False),
        "image_low_confidence_count": image_low_conf,
        "image_used_text_ratio": 0.0,
        "text_has_content": post.has_text,
        "text_sentiment": sentiment_label,
        "text_sentiment_score": round(sentiment_conf, 4),
        "text_positive_hits": int(sentiment_label == "positive"),
        "text_negative_hits": int(sentiment_label == "negative"),
        "text_topics": text_topics,
        "text_length": len(post.text or ""),
        "text_preview": (post.text or "")[:140],
        "image_detail_counts": json.dumps(detail_counts, ensure_ascii=False),
        "image_detail_confidence": json.dumps(detail_conf, ensure_ascii=False),
        "image_detail_metadata": json.dumps(detail_meta, ensure_ascii=False),
        "image_ocr_count": 0,
        "source": "gemini_mock" if mock_mode else "gemini",
        "request_pause": request_pause,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_and_write(
    classified_frames: List[pd.DataFrame],
    output_path: Path,
    enable_interactions: bool,
    quiet: bool,
) -> None:
    if not classified_frames:
        LOGGER.warning("No classified posts generated; skipping aggregation.")
        return

    posts_df = pd.concat(classified_frames, ignore_index=True)
    features = aggregate_daily_features(posts_df, enable_interactions=enable_interactions)
    if features.empty:
        LOGGER.warning("Aggregated features are empty; nothing written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    if not quiet:
        LOGGER.info("Wrote daily features to %s", output_path)


def write_classified(posts_frames: List[pd.DataFrame], path: Path, quiet: bool) -> None:
    if not posts_frames:
        return
    combined = pd.concat(posts_frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    if not quiet:
        LOGGER.info("Wrote classified posts to %s", path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    dates = list(discover_dates(args.csv_root, args.start_date, args.end_date))
    if not dates:
        raise SystemExit("No CSV files found for the specified range.")

    classifier = GeminiClassifier(
        provider=args.provider,
        gemini_model=args.gemini_model,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        max_images=args.max_images,
        mock=args.mock,
        request_pause=args.request_pause,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    classified_frames: List[pd.DataFrame] = []

    for date in dates:
        csv_path = args.csv_root / f"{date}.csv"
        if not csv_path.exists():
            LOGGER.warning("CSV missing for %s", date)
            continue

        day_df = read_daily_posts(csv_path, args.images_root)
        if args.limit:
            day_df = day_df.head(args.limit)

        classified_df = classify_posts(day_df, csv_path, classifier)
        if classified_df.empty:
            LOGGER.warning("No classifications generated for %s", date)
            continue

        classified_frames.append(classified_df)

    if args.classified_output and classified_frames:
        write_classified(classified_frames, args.classified_output, args.quiet)

    if args.dry_run:
        return 0

    aggregate_and_write(classified_frames, args.output, args.enable_interactions, args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
