"""Classify gold-related social media images into semantic buckets.

This script pairs images with their associated post text and applies zero-shot
CLIP-style classification to tag each image with one of the project-specific
categories:

* technical_chart
* news_screenshot
* gold_bullion
* meme
* irrelevant

Usage example (from repo root):

    python classify_media.py \
        --images-root images/金价 \
        --csv-root output/金价 \
        --output classified_posts.csv \
        --profile mac-cpu \
        --aggregation post

The script relies on ``open_clip`` with a hardware-aware preset (可用``--profile``覆盖). Install
requirements if needed:

    pip install open_clip_torch torch torchvision pandas pillow tqdm

输出字段会包含 ``modality``、文本情绪/主题、图像主类分布以及逐图置信度；当使用 ``--aggregation image`` 时，会保留逐张图片的分类详情。若提供 ``--image-proto-root``，脚本还会将少量人工样例图片编码为类别原型，以提升区分度。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("torch is required. Install with `pip install torch`." ) from exc

_OPEN_CLIP: Optional[Any] = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    tqdm = None  # type: ignore

LOGGER = logging.getLogger("classify_media")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_PROTOTYPES = 64

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

DEFAULT_DETAIL_TAG = {
    "technical_chart": "technical_chart.general",
    "news_screenshot": "news.general",
    "gold_bullion": "gold_bullion.product",
    "meme": "meme.general",
    "irrelevant": "noise",
}

DETAIL_PROMPTS: Dict[str, Dict[str, Sequence[str]]] = {
    "technical_chart": {
        "technical_chart.up": (
            "an upward trending financial candlestick chart",
            "股票价格持续上涨的K线图",
            "bullish chart with higher highs",
        ),
        "technical_chart.down": (
            "a downward trending candlestick chart",
            "显示价格下跌的黄金K线图",
            "bearish chart with lower lows",
        ),
        "technical_chart.sideways": (
            "a sideways consolidation candlestick chart",
            "横盘震荡的金融走势图",
            "price ranging without clear trend",
        ),
        "technical_chart.uncertain": (
            "a noisy candlestick chart without clear trend",
            "震荡且方向不明的K线图",
            "messy mixed signals trading chart",
        ),
    },
    "gold_bullion": {
        "gold_bullion.bar": (
            "stacked gold bars",
            "整齐摆放的金条",
        ),
        "gold_bullion.coin": (
            "gold coins on a table",
            "金币收藏展示",
        ),
        "gold_bullion.jewelry": (
            "gold jewelry display in a shop",
            "黄金首饰佩戴展示",
        ),
        "gold_bullion.packaging": (
            "gold gift box and packaging",
            "金店礼盒与包装袋",
        ),
    },
    "meme": {
        "meme.positive": (
            "a funny optimistic meme about making money",
            "积极搞笑的理财表情包",
            "a cheerful investing meme",
        ),
        "meme.negative": (
            "a pessimistic meme about losses",
            "抱怨亏损的投资梗图",
            "a sad investing meme",
        ),
        "meme.neutral": (
            "a neutral meme without strong emotion",
            "普通的财经梗图",
        ),
    },
}

# Default bilingual prompts for each category. Multiple phrasings improve
# zero-shot performance without any fine-tuning.
DEFAULT_CATEGORY_PROMPTS: Dict[str, Sequence[str]] = {
    "technical_chart": (
        "a detailed candlestick chart for gold price trend",
        "a tradingview screenshot with technical indicators",
        "a financial market line chart with candles",
        "黄金价格的K线走势图",
        "包含MACD或均线指标的黄金技术分析图",
    ),
    "news_screenshot": (
        "a cropped screenshot of a financial news website",
        "a TV news overlay with ticker text about gold",
        "headline and paragraphs describing gold price",
        "财经新闻播报画面",
        "包含大量文字段落的黄金新闻截图",
    ),
    "gold_bullion": (
        "a product photo of stacked gold bars",
        "a close-up shot of gold jewelry in a store",
        "pure gold coins on a reflective surface",
        "实物金条或金币的高清照片",
        "珠宝店里陈列的黄金首饰",
    ),
    "meme": (
        "a humorous meme about finance with bold captions",
        "colorful cartoon style joke about gold price",
        "an internet joke image with big top text",
        "带文字说明的搞笑表情包",
        "网络梗图或段子配图",
    ),
    "irrelevant": (
        "a random lifestyle photo unrelated to finance",
        "a scenic photo or portrait not connected to gold",
        "daily life snapshot with no financial content",
        "与黄金或财经无关的普通风景照",
        "日常人物照片，未体现金融信息",
    ),
}


@dataclass
class TextAnalysis:
    raw_text: str
    has_text: bool
    sentiment: str
    sentiment_score: float
    positive_hits: List[str] = field(default_factory=list)
    negative_hits: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    length: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text_has_content": self.has_text,
            "text_sentiment": self.sentiment,
            "text_sentiment_score": self.sentiment_score,
            "text_positive_hits": ";".join(self.positive_hits),
            "text_negative_hits": ";".join(self.negative_hits),
            "text_topics": ";".join(self.topics),
            "text_length": self.length,
            "text_preview": self.raw_text[:120],
        }


@dataclass
class PostRecord:
    csv_path: Path
    post_id: str
    text: str
    day: Optional[str]
    row_index: int
    image_paths: List[Path] = field(default_factory=list)
    text_analysis: Optional[TextAnalysis] = None
    modality: Optional[str] = None

    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    def has_image(self) -> bool:
        return bool(self.image_paths)


@dataclass
class DataPoint:
    post: PostRecord
    image_path: Path


@dataclass
class ClassificationResult:
    image_path: Path
    csv_path: Path
    post_id: str
    assigned_category: str
    confidence: float
    ranked_scores: Dict[str, float]
    confidence_gap: float
    second_best: Optional[str]
    low_confidence: bool
    used_text: bool
    day: Optional[str]
    row_index: int
    post: PostRecord
    detail_tag: str
    detail_scores: Dict[str, Any]
    ocr_text: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_images_root = script_dir / "images" / "金价"
    default_csv_root = script_dir / "output" / "金价"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-root",
        type=Path,
        default=default_images_root,
        help="Root directory that contains dated folders with images.",
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=default_csv_root,
        help="Directory holding daily CSV files with metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("classified_images.csv"),
        help="Where to write the classification results (CSV).",
    )
    parser.add_argument(
        "--aggregation",
        choices=["post", "image"],
        default="post",
        help="Aggregation level of the output CSV (post-level summary or per-image records).",
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "mac-cpu", "mac-mps", "gpu-server"],
        default="auto",
        help="Quick hardware preset to tune device/model/batch size.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="open_clip model architecture to use (overrides --profile).",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Which pretrained weights to load (defaults depend on profile).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device spec; 'auto' selects CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Images per batch. Leave unset to auto-tune from --profile/device.",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.35,
        help="Relative weight for the post text embedding when fusing features.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images to process (useful for quick tests).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Restrict processing to a specific YYYY-MM-DD folder and CSV.",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=None,
        help="Optional JSON file that maps category names to prompt lists.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="低于此置信度会在结果中打上low_confidence标记。",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.03,
        help="Top1与Top2分数差值低于该阈值时同样标记为low_confidence。",
    )
    parser.add_argument(
        "--image-proto-root",
        type=Path,
        default=None,
        help="(可选) 分类前先读取每个类别的样例图片，目录结构需按类别分子文件夹。",
    )
    parser.add_argument(
        "--proto-text-weight",
        type=float,
        default=1.0,
        help="构建类别原型时文本提示的权重。",
    )
    parser.add_argument(
        "--proto-image-weight",
        type=float,
        default=1.0,
        help="构建类别原型时样例图片特征的权重。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matches that would be processed without running the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def expand_prompts(prompts: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for base in prompts:
        base = base.strip()
        if not base:
            continue
        expanded.append(base)
        contains_non_ascii = any(ord(ch) > 127 for ch in base)
        if contains_non_ascii:
            expanded.append(f"一张展示{base}的图片")
            expanded.append(f"与{base}相关的照片")
        else:
            expanded.append(f"a photo of {base}")
            expanded.append(f"an image showing {base}")
    # 去重同时保持顺序
    return list(dict.fromkeys(expanded))


def resolve_device(device_arg: str, profile: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if profile == "mac-mps":
        if torch.backends.mps.is_available():  # pragma: no branch - macOS path
            return torch.device("mps")
        LOGGER.warning("选择了mac-mps，但当前PyTorch未检测到MPS，改为CPU")
        return torch.device("cpu")

    if profile == "mac-cpu":
        return torch.device("cpu")

    if profile == "gpu-server" and torch.cuda.is_available():
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - macOS path
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_and_pretrained(
    model_arg: Optional[str],
    pretrained_arg: Optional[str],
    profile: str,
    device: torch.device,
) -> Tuple[str, str]:
    if model_arg:
        return model_arg, pretrained_arg or "openai"

    pretrained = pretrained_arg or "openai"

    if profile == "mac-cpu":
        return "ViT-B-32", pretrained
    if profile == "mac-mps":
        return "ViT-B-16", pretrained
    if profile == "gpu-server" and device.type == "cuda":
        return "ViT-L-14", pretrained

    if device.type == "cuda":
        return "ViT-L-14", pretrained
    if device.type == "mps":  # pragma: no cover - macOS path
        return "ViT-B-16", pretrained
    return "ViT-B-32", pretrained


def resolve_batch_size(
    configured: Optional[int],
    profile: str,
    device: torch.device,
) -> int:
    if configured is not None and configured > 0:
        return configured

    if profile == "mac-cpu":
        return 4
    if profile == "mac-mps":
        return 8
    if profile == "gpu-server":
        return 32

    if device.type == "cuda":
        return 32
    if device.type == "mps":  # pragma: no cover - macOS path
        return 8
    return 4


def normalize_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(value)
    result = str(value).strip()
    if result.lower() in {"nan", "none", "null"}:
        return ""
    return result


def determine_modality_flags(has_text: bool, has_image: bool) -> str:
    if has_text and has_image:
        return "text_image"
    if has_text:
        return "text_only"
    if has_image:
        return "image_only"
    return "empty"


def determine_modality(post: PostRecord) -> str:
    return determine_modality_flags(post.has_text(), post.has_image())


def analyze_text(text: str) -> TextAnalysis:
    cleaned = text.strip() if text else ""
    if not cleaned:
        return TextAnalysis(
            raw_text="",
            has_text=False,
            sentiment="none",
            sentiment_score=0.0,
            topics=[],
            positive_hits=[],
            negative_hits=[],
            length=0,
        )

    pos_hits = [word for word in POSITIVE_TERMS if word in cleaned]
    neg_hits = [word for word in NEGATIVE_TERMS if word in cleaned]
    score = float(len(pos_hits) - len(neg_hits))
    if score > 0:
        sentiment = "positive"
    elif score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    topics: List[str] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in cleaned for keyword in keywords):
            topics.append(topic)

    return TextAnalysis(
        raw_text=cleaned,
        has_text=True,
        sentiment=sentiment,
        sentiment_score=score,
        positive_hits=pos_hits,
        negative_hits=neg_hits,
        topics=topics,
        length=len(cleaned),
    )


def perform_ocr(image_path: Path) -> str:
    if pytesseract is None:
        return ""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
        text = pytesseract.image_to_string(gray, lang="chi_sim+eng")
        return text.strip()
    except Exception as exc:  # pragma: no cover - OCR is best-effort
        LOGGER.debug("OCR failed for %s: %s", image_path, exc)
        return ""


def analysis_to_brief_dict(analysis: TextAnalysis) -> Dict[str, Any]:
    return {
        "sentiment": analysis.sentiment,
        "sentiment_score": analysis.sentiment_score,
        "positive_hits": ";".join(analysis.positive_hits),
        "negative_hits": ";".join(analysis.negative_hits),
        "topics": ";".join(analysis.topics),
        "length": analysis.length,
    }


def select_detail_from_prototypes(
    category: str,
    feature: torch.Tensor,
    detail_prototypes: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[str, Dict[str, Any]]:
    bucket = detail_prototypes.get(category)
    if not bucket:
        return DEFAULT_DETAIL_TAG.get(category, ""), {}
    sims: Dict[str, float] = {}
    for key, proto in bucket.items():
        sims[key] = float((feature @ proto).item())
    best_key = max(sims, key=sims.get)
    rounded = {k: round(v, 4) for k, v in sims.items()}
    return best_key, {"detail_similarities": rounded}


def merge_sentiment_detail(prefix: str, analysis: TextAnalysis) -> Tuple[str, Dict[str, Any]]:
    detail_tag = f"{prefix}.{analysis.sentiment}"
    detail_scores = analysis_to_brief_dict(analysis)
    return detail_tag, detail_scores


def determine_detail_for_image(
    category: str,
    feature: torch.Tensor,
    dp: DataPoint,
    detail_prototypes: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[str, Dict[str, Any], str]:
    detail_tag = DEFAULT_DETAIL_TAG.get(category, "")
    detail_scores: Dict[str, Any] = {}
    ocr_text = ""

    if category == "news_screenshot":
        ocr_text = perform_ocr(dp.image_path)
        source_text = ocr_text or dp.post.text
        analysis = analyze_text(source_text)
        detail_tag, detail_scores = merge_sentiment_detail("news", analysis)
        detail_scores["sentiment_source"] = "ocr" if ocr_text else "post"
        detail_scores["ocr_length"] = len(ocr_text)
    elif category == "technical_chart":
        detail_tag, detail_scores = select_detail_from_prototypes(
            category, feature, detail_prototypes
        )
        if detail_scores:
            detail_scores["source"] = "clip"
    elif category == "gold_bullion":
        detail_tag, detail_scores = select_detail_from_prototypes(
            category, feature, detail_prototypes
        )
        if detail_scores:
            detail_scores["source"] = "clip"
    elif category == "meme":
        ocr_text = perform_ocr(dp.image_path)
        combined_text = "\n".join(filter(None, [dp.post.text, ocr_text]))
        analysis = analyze_text(combined_text)
        detail_tag, detail_scores = merge_sentiment_detail("meme", analysis)
        detail_scores["sentiment_source"] = "text+ocr" if ocr_text else "post"
        detail_scores["ocr_length"] = len(ocr_text)
        clip_tag, clip_scores = select_detail_from_prototypes(
            category, feature, detail_prototypes
        )
        if clip_scores:
            detail_scores["clip_detail"] = clip_tag
            detail_scores["clip_scores"] = clip_scores.get("detail_similarities")
    else:
        if category in detail_prototypes:
            detail_tag, detail_scores = select_detail_from_prototypes(
                category, feature, detail_prototypes
            )

    if not detail_tag:
        detail_tag = DEFAULT_DETAIL_TAG.get(category, "")

    return detail_tag, detail_scores, ocr_text


def prepare_posts(args: argparse.Namespace) -> Tuple[
    Dict[str, Sequence[str]],
    List[Path],
    List[PostRecord],
    List[DataPoint],
    Dict[str, int],
]:
    categories = load_category_prompts(args.categories)
    csv_files = find_csv_files(args.csv_root, args.date)

    post_limit = args.limit if args.aggregation == "post" else None
    posts = collect_posts(csv_files, post_limit)

    for post in posts:
        post.text_analysis = analyze_text(post.text)
        post.modality = determine_modality(post)

    image_limit = args.limit if args.aggregation == "image" else None
    datapoints = build_image_datapoints(posts, image_limit)

    stats = {
        "csv_files": len(csv_files),
        "posts": len(posts),
        "posts_with_images": sum(1 for post in posts if post.has_image()),
        "images": len(datapoints),
    }

    return categories, csv_files, posts, datapoints, stats


def run_classification(args: argparse.Namespace) -> Tuple[
    Dict[str, int],
    List[PostRecord],
    List[DataPoint],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    categories, csv_files, posts, datapoints, stats = prepare_posts(args)

    if args.dry_run:
        return stats, posts, datapoints, None, None

    image_results: List[ClassificationResult] = []
    detail_prototypes: Dict[str, Dict[str, torch.Tensor]] = {}

    if datapoints:
        device = resolve_device(args.device, args.profile)
        model_name, pretrained = resolve_model_and_pretrained(
            args.model, args.pretrained, args.profile, device
        )
        batch_size = resolve_batch_size(args.batch_size, args.profile, device)

        LOGGER.info("使用计算设备: %s", device)
        LOGGER.info("加载模型: %s (%s), batch_size=%d", model_name, pretrained, batch_size)

        model, preprocess, tokenizer = load_model_and_tokenizer(model_name, pretrained, device)
        prototypes, detail_prototypes = encode_category_prototypes(
            model=model,
            tokenizer=tokenizer,
            categories=categories,
            device=device,
            preprocess=preprocess,
            image_proto_root=args.image_proto_root,
            proto_text_weight=args.proto_text_weight,
            proto_image_weight=args.proto_image_weight,
        )
        image_results = classify_datapoints(
            datapoints=datapoints,
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=device,
            prototypes=prototypes,
            detail_prototypes=detail_prototypes,
            text_weight=args.text_weight,
            batch_size=batch_size,
            min_confidence=args.min_confidence,
            min_gap=args.min_gap,
        )
        images_df = results_to_dataframe(image_results)
    else:
        images_df = results_to_dataframe([])

    posts_df = aggregate_posts_to_dataframe(posts, image_results)

    return stats, posts, datapoints, posts_df, images_df


def load_category_prompts(custom: Optional[Path]) -> Dict[str, Sequence[str]]:
    if custom is None:
        return DEFAULT_CATEGORY_PROMPTS
    with custom.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Custom categories file must map category to prompt list")
    normalized: Dict[str, Sequence[str]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            normalized[key] = [value]
        elif isinstance(value, Sequence):
            normalized[key] = [str(item) for item in value]
        else:
            raise ValueError(f"Prompts for {key} must be string or sequence")
    return normalized


def find_csv_files(csv_root: Path, date_filter: Optional[str]) -> List[Path]:
    if not csv_root.exists():
        raise FileNotFoundError(f"CSV root not found: {csv_root}")
    candidates = sorted(p for p in csv_root.glob("*.csv") if p.is_file())
    if date_filter:
        suffix = f"{date_filter}.csv"
        candidates = [p for p in candidates if p.name.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError("No CSV files matched the criteria.")
    return candidates


def split_image_paths(value: str) -> Iterable[str]:
    for part in value.split(";"):
        trimmed = part.strip()
        if trimmed:
            yield trimmed


def collect_posts(csv_paths: Sequence[Path], limit: Optional[int]) -> List[PostRecord]:
    posts: List[PostRecord] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            text = normalize_str(row.get("text"))
            post_id = normalize_str(row.get("post_id")) or f"{csv_path.stem}-{idx}"
            day_value = normalize_str(row.get("day")) or None
            image_field = normalize_str(row.get("image_paths"))
            image_paths = [Path(path) for path in split_image_paths(image_field)] if image_field else []

            post = PostRecord(
                csv_path=csv_path,
                post_id=post_id,
                text=text,
                day=day_value,
                row_index=int(idx),
                image_paths=image_paths,
            )
            posts.append(post)
            if limit is not None and len(posts) >= limit:
                return posts
    if not posts:
        raise RuntimeError("No posts discovered from the provided CSV files.")
    return posts


def build_image_datapoints(posts: Sequence[PostRecord], limit: Optional[int]) -> List[DataPoint]:
    datapoints: List[DataPoint] = []
    for post in posts:
        for image_path in post.image_paths:
            datapoints.append(DataPoint(post=post, image_path=image_path))
            if limit is not None and len(datapoints) >= limit:
                return datapoints
    return datapoints


def get_open_clip_module():
    global _OPEN_CLIP
    if _OPEN_CLIP is None:
        try:
            import open_clip as imported_open_clip  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "open_clip_torch is required. Install with `pip install open_clip_torch`."
            ) from exc
        _OPEN_CLIP = imported_open_clip
    return _OPEN_CLIP


def load_model_and_tokenizer(model_name: str, pretrained: str, device: torch.device):
    open_clip = get_open_clip_module()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, preprocess, tokenizer


def encode_category_prototypes(
    model,
    tokenizer,
    categories: Dict[str, Sequence[str]],
    device: torch.device,
    preprocess,
    image_proto_root: Optional[Path],
    proto_text_weight: float,
    proto_image_weight: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    text_prototypes: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, prompts in categories.items():
            prompt_list = expand_prompts(prompts)
            if not prompt_list:
                continue
            tokens = tokenizer(prompt_list).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            proto = features.mean(dim=0)
            proto = proto / proto.norm()
            text_prototypes[name] = proto

    image_prototypes: Dict[str, torch.Tensor] = {}
    if image_proto_root is not None:
        for name in categories.keys():
            category_dir = image_proto_root / name
            if not category_dir.exists():
                continue
            image_tensors: List[torch.Tensor] = []
            count = 0
            for path in sorted(category_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(path) as img:
                        tensor = preprocess(img.convert("RGB"))
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning("样例图片读取失败 %s: %s", path, exc)
                    continue
                image_tensors.append(tensor)
                count += 1
                if count >= MAX_IMAGE_PROTOTYPES:
                    break
            if not image_tensors:
                continue
            batch = torch.stack(image_tensors).to(device)
            with torch.no_grad():
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            proto = features.mean(dim=0)
            proto = proto / proto.norm()
            image_prototypes[name] = proto
            LOGGER.info("类别 %s : 加载了 %d 张样例图片用于原型", name, len(image_tensors))

    detail_prototypes: Dict[str, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for category, detail_map in DETAIL_PROMPTS.items():
            proto_bucket: Dict[str, torch.Tensor] = {}
            for detail_key, prompts in detail_map.items():
                prompt_list = expand_prompts(prompts)
                if not prompt_list:
                    continue
                tokens = tokenizer(prompt_list).to(device)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                proto = features.mean(dim=0)
                proto = proto / proto.norm()
                proto_bucket[detail_key] = proto
            if proto_bucket:
                detail_prototypes[category] = proto_bucket

    prototypes: Dict[str, torch.Tensor] = {}
    for name in categories.keys():
        components: List[torch.Tensor] = []
        if name in text_prototypes and proto_text_weight > 0:
            components.append(proto_text_weight * text_prototypes[name])
        if name in image_prototypes and proto_image_weight > 0:
            components.append(proto_image_weight * image_prototypes[name])

        if not components:
            if name in text_prototypes:
                components.append(text_prototypes[name])
            elif name in image_prototypes:
                components.append(image_prototypes[name])
            else:
                raise RuntimeError(f"类别 {name} 缺少原型，请检查提示词或样例路径")

        combined = torch.stack(components).sum(dim=0)
        combined = combined / combined.norm()
        prototypes[name] = combined

    return prototypes, detail_prototypes


def image_to_tensor(image_path: Path, preprocess) -> torch.Tensor:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return preprocess(image)


def should_use_text(dp: DataPoint) -> bool:
    post = dp.post
    if not post.post_id:
        return False
    if not post.has_text():
        return False
    filename = dp.image_path.name
    if not filename:
        return False
    if post.post_id in filename:
        return True
    stem = dp.image_path.stem
    return bool(stem and stem.startswith(post.post_id))


def encode_batch(
    model,
    tokenizer,
    batch: Sequence[DataPoint],
    preprocess,
    device: torch.device,
    text_weight: float,
) -> Tuple[List[torch.Tensor], List[int], List[bool]]:
    images: List[torch.Tensor] = []
    valid_indices: List[int] = []
    text_map: Dict[int, torch.Tensor] = {}

    for idx, dp in enumerate(batch):
        if not dp.image_path.exists():
            LOGGER.warning("Image not found: %s", dp.image_path)
            continue
        try:
            tensor = image_to_tensor(dp.image_path, preprocess)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to load %s: %s", dp.image_path, exc)
            continue
        images.append(tensor)
        valid_indices.append(idx)

    if not images:
        return [], [], []

    image_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Encode associated text (if any)
    texts_to_encode: List[str] = []
    text_slots: List[int] = []
    for pos, dp in enumerate(batch):
        if pos not in valid_indices:
            continue
        if not should_use_text(dp):
            continue
        txt = (dp.post.text or "").strip()
        if not txt:
            continue
        texts_to_encode.append(txt)
        text_slots.append(pos)

    if texts_to_encode:
        tokens = tokenizer(texts_to_encode).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for slot, feat in zip(text_slots, text_features):
            text_map[slot] = feat

    combined_features: List[torch.Tensor] = []
    used_text_flags: List[bool] = []
    for feature, pos in zip(image_features, valid_indices):
        text_feat = text_map.get(pos)
        if text_feat is not None and text_weight > 0:
            fused = feature + text_weight * text_feat
            fused = fused / fused.norm()
            combined_features.append(fused)
            used_text_flags.append(True)
        else:
            combined_features.append(feature)
            used_text_flags.append(False)

    return combined_features, valid_indices, used_text_flags


def classify_datapoints(
    datapoints: Sequence[DataPoint],
    model,
    tokenizer,
    preprocess,
    device: torch.device,
    prototypes: Dict[str, torch.Tensor],
    detail_prototypes: Dict[str, Dict[str, torch.Tensor]],
    text_weight: float,
    batch_size: int,
    min_confidence: float,
    min_gap: float,
) -> List[ClassificationResult]:
    prototype_names = list(prototypes.keys())
    prototype_stack = torch.stack([prototypes[name] for name in prototype_names]).to(device)

    results: List[ClassificationResult] = []
    iterator = range(0, len(datapoints), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="classifying", total=(len(datapoints) + batch_size - 1) // batch_size)

    for start in iterator:
        batch = datapoints[start : start + batch_size]
        combined_features, valid_indices, used_text_flags = encode_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            preprocess=preprocess,
            device=device,
            text_weight=text_weight,
        )
        if not combined_features:
            continue
        feature_stack = torch.stack(combined_features)
        similarities = feature_stack @ prototype_stack.T

        for local_idx, sim_vector in enumerate(similarities):
            dp = batch[valid_indices[local_idx]]
            feature = combined_features[local_idx]
            scores = sim_vector.detach().cpu()
            ranked = {name: float(scores[i].item()) for i, name in enumerate(prototype_names)}

            topk_values, topk_indices = torch.topk(scores, k=min(2, len(prototype_names)))
            best_idx = int(topk_indices[0].item())
            best_score = float(topk_values[0].item())
            second_best_score: float
            second_best_name: Optional[str]
            if len(topk_values) > 1:
                second_best_score = float(topk_values[1].item())
                second_best_name = prototype_names[int(topk_indices[1].item())]
            else:  # pragma: no cover - single class edge case
                second_best_score = -1.0
                second_best_name = None
            gap = best_score - second_best_score if second_best_score >= 0 else 1.0
            low_confidence = bool(best_score < min_confidence or gap < min_gap)

            detail_tag, detail_scores, ocr_text = determine_detail_for_image(
                prototype_names[best_idx], feature, dp, detail_prototypes
            )

            results.append(
                ClassificationResult(
                    image_path=dp.image_path,
                    csv_path=dp.post.csv_path,
                    post_id=dp.post.post_id,
                    assigned_category=prototype_names[best_idx],
                    confidence=best_score,
                    ranked_scores=ranked,
                    confidence_gap=gap,
                    second_best=second_best_name,
                    low_confidence=low_confidence,
                    used_text=used_text_flags[local_idx],
                    day=dp.post.day,
                    row_index=dp.post.row_index,
                    post=dp.post,
                    detail_tag=detail_tag,
                    detail_scores=detail_scores,
                    ocr_text=ocr_text,
                )
            )

    return results


def results_to_dataframe(results: Sequence[ClassificationResult]) -> pd.DataFrame:
    records = []
    for res in results:
        post = res.post
        text_info = post.text_analysis or analyze_text(post.text)
        modality = post.modality or determine_modality(post)
        records.append(
            {
                "image_path": str(res.image_path),
                "csv_path": str(res.csv_path),
                "post_id": res.post_id,
                "day": res.day,
                "row_index": res.row_index,
                "modality": modality,
                "has_text": post.has_text(),
                "has_image": post.has_image(),
                "category": res.assigned_category,
                "confidence": res.confidence,
                "second_best": res.second_best,
                "confidence_gap": res.confidence_gap,
                "low_confidence": res.low_confidence,
                "used_text": res.used_text,
                "images_in_post": len(post.image_paths),
                "detail_tag": res.detail_tag,
                "detail_scores": json.dumps(res.detail_scores, ensure_ascii=False),
                "ocr_text_preview": res.ocr_text[:160],
                "ocr_text_length": len(res.ocr_text),
                **text_info.as_dict(),
                "scores": json.dumps(res.ranked_scores, ensure_ascii=False),
            }
        )
    return pd.DataFrame.from_records(records)


def summarise_image_results(results: Sequence[ClassificationResult]) -> Dict[str, Any]:
    if not results:
        return {
            "image_main_category": "",
            "image_detail_tag": "",
            "image_main_confidence": 0.0,
            "images_classified": 0,
            "image_category_counts": json.dumps({}, ensure_ascii=False),
            "image_category_confidence": json.dumps({}, ensure_ascii=False),
            "image_low_confidence_count": 0,
            "image_used_text_ratio": 0.0,
        }

    count_by_cat: Dict[str, int] = defaultdict(int)
    confidence_sum: Dict[str, float] = defaultdict(float)
    detail_count: Dict[str, int] = defaultdict(int)
    detail_confidence_sum: Dict[str, float] = defaultdict(float)
    detail_meta: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    ocr_texts: List[str] = []
    low_confidence_count = 0
    used_text_count = 0

    for res in results:
        cat = res.assigned_category
        count_by_cat[cat] += 1
        confidence_sum[cat] += res.confidence
        if res.low_confidence:
            low_confidence_count += 1
        if res.used_text:
            used_text_count += 1
        if res.detail_tag:
            detail_count[res.detail_tag] += 1
            detail_confidence_sum[res.detail_tag] += res.confidence
            if res.detail_scores:
                detail_meta[res.detail_tag].append(res.detail_scores)
        if res.ocr_text:
            ocr_texts.append(res.ocr_text)

    def score_key(item: Tuple[str, float]) -> Tuple[float, int]:
        cat, score_sum = item
        return (score_sum, count_by_cat.get(cat, 0))

    best_cat = max(confidence_sum.items(), key=score_key)[0]
    avg_confidence = {
        cat: confidence_sum[cat] / count_by_cat[cat]
        for cat in count_by_cat
        if count_by_cat[cat] > 0
    }

    detail_tag = DEFAULT_DETAIL_TAG.get(best_cat, "")

    avg_detail_confidence = {
        tag: detail_confidence_sum[tag] / detail_count[tag]
        for tag in detail_count
        if detail_count[tag] > 0
    }

    best_detail = detail_tag
    detail_candidates = {
        tag: detail_count[tag]
        for tag in detail_count
        if tag.startswith(best_cat)
    }
    if detail_candidates:
        best_detail = max(
            detail_candidates.items(),
            key=lambda item: (item[1], avg_detail_confidence.get(item[0], 0.0)),
        )[0]

    return {
        "image_main_category": best_cat,
        "image_detail_tag": best_detail,
        "image_main_confidence": round(avg_confidence.get(best_cat, 0.0), 4),
        "images_classified": len(results),
        "image_category_counts": json.dumps(dict(count_by_cat), ensure_ascii=False),
        "image_category_confidence": json.dumps({cat: round(avg_confidence.get(cat, 0.0), 4) for cat in avg_confidence}, ensure_ascii=False),
        "image_low_confidence_count": low_confidence_count,
        "image_used_text_ratio": round(used_text_count / len(results), 4),
        "image_detail_counts": json.dumps(dict(detail_count), ensure_ascii=False),
        "image_detail_confidence": json.dumps(
            {tag: round(avg_detail_confidence.get(tag, 0.0), 4) for tag in avg_detail_confidence},
            ensure_ascii=False,
        ),
        "image_detail_metadata": json.dumps(
            {
                tag: {
                    "count": detail_count[tag],
                    "avg_confidence": round(avg_detail_confidence.get(tag, 0.0), 4),
                    "examples": detail_meta[tag][:3],
                }
                for tag in detail_meta
            },
            ensure_ascii=False,
        ),
        "image_ocr_preview": " | ".join(text[:80] for text in ocr_texts[:3]),
        "image_ocr_count": len(ocr_texts),
    }


def aggregate_posts_to_dataframe(
    posts: Sequence[PostRecord],
    image_results: Sequence[ClassificationResult],
) -> pd.DataFrame:
    grouped: Dict[int, List[ClassificationResult]] = defaultdict(list)
    for res in image_results:
        grouped[id(res.post)].append(res)

    records = []
    for post in posts:
        text_info = post.text_analysis or analyze_text(post.text)
        modality = post.modality or determine_modality(post)
        image_summary = summarise_image_results(grouped.get(id(post), []))
        record = {
            "csv_path": str(post.csv_path),
            "post_id": post.post_id,
            "day": post.day,
            "row_index": post.row_index,
            "modality": modality,
            "has_text": post.has_text(),
            "has_image": post.has_image(),
            "images_total": len(post.image_paths),
            **image_summary,
            **text_info.as_dict(),
        }
        records.append(record)

    if not records:
        raise RuntimeError("No posts available for aggregation.")

    return pd.DataFrame.from_records(records)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    stats, posts, datapoints, posts_df, images_df = run_classification(args)

    LOGGER.info(
        "Loaded %d posts (%d with images) from %d CSV files; total images queued=%d",
        stats["posts"],
        stats["posts_with_images"],
        stats["csv_files"],
        stats["images"],
    )

    if args.dry_run:
        preview_count = min(stats["posts"], args.limit or 10)
        for post in posts[:preview_count]:
            text_info = post.text_analysis or analyze_text(post.text)
            LOGGER.info(
                "Post %s modality=%s images=%d sentiment=%s topics=%s",
                post.post_id,
                post.modality,
                len(post.image_paths),
                text_info.sentiment,
                ",".join(text_info.topics),
            )
        if args.aggregation == "image":
            for dp in datapoints[: args.limit or 20]:
                LOGGER.info(
                    "Would process image %s (post %s)",
                    dp.image_path,
                    dp.post.post_id,
                )
        LOGGER.info("Dry run complete; exiting without classification.")
        return 0

    if args.aggregation == "image":
        if images_df is None or images_df.empty:
            raise RuntimeError("Image aggregation requested but no image results were produced.")
        df = images_df
    else:
        if posts_df is None:
            raise RuntimeError("Post aggregation failed to produce results.")
        df = posts_df
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
