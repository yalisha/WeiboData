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
        --output classified_gold_images.csv \
        --profile mac-cpu

The script relies on ``open_clip`` with a hardware-aware preset (可用``--profile``覆盖). Install
requirements if needed:

    pip install open_clip_torch torch torchvision pandas pillow tqdm

输出字段会包含 ``second_best``、``confidence_gap``、``low_confidence``，以及 ``used_text``（仅当图片文件名包含对应 ``post_id`` 时才会融合文本特征）。若提供 ``--image-proto-root``，脚本还会将少量人工样例图片编码为类别原型，以提升区分度。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

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
class DataPoint:
    image_path: Path
    csv_path: Path
    post_id: str
    text: str
    day: Optional[str]
    row_index: int


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


def collect_datapoints(csv_paths: Sequence[Path], limit: Optional[int]) -> List[DataPoint]:
    datapoints: List[DataPoint] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            image_field = str(row.get("image_paths", "") or "").strip()
            if not image_field:
                continue
            post_id = str(row.get("post_id", ""))
            text = str(row.get("text", "") or "")
            day = str(row.get("day") or "") or None
            for raw_path in split_image_paths(image_field):
                image_path = Path(raw_path)
                datapoints.append(
                    DataPoint(
                        image_path=image_path,
                        csv_path=csv_path,
                        post_id=post_id,
                        text=text,
                        day=day,
                        row_index=int(idx),
                    )
                )
                if limit is not None and len(datapoints) >= limit:
                    return datapoints
    if not datapoints:
        raise RuntimeError("No images discovered from the provided CSV files.")
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
) -> Dict[str, torch.Tensor]:
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

    return prototypes


def image_to_tensor(image_path: Path, preprocess) -> torch.Tensor:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    return preprocess(image)


def should_use_text(dp: DataPoint) -> bool:
    if not dp.post_id:
        return False
    if not (dp.text and dp.text.strip()):
        return False
    filename = dp.image_path.name
    if not filename:
        return False
    if dp.post_id in filename:
        return True
    stem = dp.image_path.stem
    return bool(stem and stem.startswith(dp.post_id))


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
        txt = (dp.text or "").strip()
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

            results.append(
                ClassificationResult(
                    image_path=dp.image_path,
                    csv_path=dp.csv_path,
                    post_id=dp.post_id,
                    assigned_category=prototype_names[best_idx],
                    confidence=best_score,
                    ranked_scores=ranked,
                    confidence_gap=gap,
                    second_best=second_best_name,
                    low_confidence=low_confidence,
                    used_text=used_text_flags[local_idx],
                    day=dp.day,
                    row_index=dp.row_index,
                )
            )

    return results


def results_to_dataframe(results: Sequence[ClassificationResult]) -> pd.DataFrame:
    if not results:
        raise RuntimeError("No classification results produced.")
    records = []
    for res in results:
        records.append(
            {
                "image_path": str(res.image_path),
                "csv_path": str(res.csv_path),
                "post_id": res.post_id,
                "day": res.day,
                "row_index": res.row_index,
                "category": res.assigned_category,
                "confidence": res.confidence,
                "second_best": res.second_best,
                "confidence_gap": res.confidence_gap,
                "low_confidence": res.low_confidence,
                "used_text": res.used_text,
                "scores": json.dumps(res.ranked_scores, ensure_ascii=False),
            }
        )
    return pd.DataFrame.from_records(records)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    categories = load_category_prompts(args.categories)
    csv_files = find_csv_files(args.csv_root, args.date)
    datapoints = collect_datapoints(csv_files, args.limit)

    LOGGER.info("Discovered %d images from %d CSV files", len(datapoints), len(csv_files))

    if args.dry_run:
        for dp in datapoints[: args.limit or 20]:
            LOGGER.info("Would process %s (post %s)", dp.image_path, dp.post_id)
        LOGGER.info("Dry run complete; exiting without classification.")
        return 0

    device = resolve_device(args.device, args.profile)
    model_name, pretrained = resolve_model_and_pretrained(
        args.model, args.pretrained, args.profile, device
    )
    batch_size = resolve_batch_size(args.batch_size, args.profile, device)

    LOGGER.info("使用计算设备: %s", device)
    LOGGER.info("加载模型: %s (%s), batch_size=%d", model_name, pretrained, batch_size)

    model, preprocess, tokenizer = load_model_and_tokenizer(model_name, pretrained, device)
    prototypes = encode_category_prototypes(
        model=model,
        tokenizer=tokenizer,
        categories=categories,
        device=device,
        preprocess=preprocess,
        image_proto_root=args.image_proto_root,
        proto_text_weight=args.proto_text_weight,
        proto_image_weight=args.proto_image_weight,
    )
    results = classify_datapoints(
        datapoints=datapoints,
        model=model,
        tokenizer=tokenizer,
        preprocess=preprocess,
        device=device,
        prototypes=prototypes,
        text_weight=args.text_weight,
        batch_size=batch_size,
        min_confidence=args.min_confidence,
        min_gap=args.min_gap,
    )

    df = results_to_dataframe(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    LOGGER.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
