## 数据概况（阶段 1）

- 使用 `summarize_modality.py output/金价` 统计 303,091 条帖子；其中文本+图片 163,196 条、纯文本 139,895 条、纯图片为 0、空帖子为 0。
- 按日粒度的明细已写入 `modality_stats.csv`，后续细分类与特征工程可直接复用。
- 统计逻辑基于 `text` 字段是否为空以及 `image_paths` 是否存在（分号分隔的图片路径）。如需限定日期，可使用 `--limit` 或自行筛选。

## 阶段 2：图文聚合分类管线

- `classify_media.py` 新增 `--aggregation {post,image}`（默认 `post`）：
  - `post` 会为每条帖子输出统一记录，包含模态（text_only/image_only/text_image）、图像主类/细类占比、文本情绪（基于正负面词典）与主题标签。
  - `image` 保留逐图调试视角，在每行记录里补充所属帖子的模态、文本情绪以及原有的置信度信息。
- 帖子加载阶段自动融合文本：
  - 文本情绪使用内置中英文关键词计算（可继续补充词典）；`text_sentiment_score` 为正负命中差值，`text_topics` 根据关键词打上 `macro`、`technical`、`jewelry`、`meme_text`、`risk` 等标签。
  - `modality` 字段区分纯文本、纯图、图文；`text_has_content`/`has_image` 便于统计缺失情况。
- 图像分类阶段保持原零样本+原型增强流程。聚合逻辑会统计每类数量/平均置信度、低置信度张数、文本特征使用率，并输出默认细类标签（后续阶段继续细化）。
- 样例命令：
  ```bash
  # 帖子级聚合
  python classify_media.py \
      --profile mac-cpu \
      --date 2022-01-01 \
      --image-proto-root prototypes \
      --aggregation post \
      --output classified_posts.csv

  # 调试逐图结果
  python classify_media.py \
      --profile mac-cpu \
      --date 2022-01-01 \
      --aggregation image \
      --limit 20 \
      --dry-run
  ```

## 阶段 3：细粒度标注逻辑

- `technical_chart`：基于 CLIP 细类原型计算趋势标签（`up/down/sideways/uncertain`），在帖子级输出 `image_detail_tag`、`image_detail_counts`、`image_detail_confidence` 等统计。
- `news_screenshot`：调用可选 `pytesseract` 做 OCR，结合识别到的文本或原帖文字进行情绪判定，生成 `news.positive/negative/neutral` 等细类，同时保留 `ocr_text_preview`。
- `gold_bullion`：新增金条/金币/首饰/包装原型，按相似度自动细分产品类型。
- `meme`：综合 OCR 文本与帖子文本进行情绪分析（`meme.positive/negative/neutral`），并记录 CLIP 细类得分辅助人工复核。
- 每张图片的细粒度信息通过 `detail_tag`、`detail_scores`、`ocr_text_*` 字段输出；帖子级 CSV 会聚合细类计数、平均置信度以及 OCR 预览，方便后续特征抽取。

## 阶段 4：结果汇总与验证

- `batch_classify.py` 封装批量流程：
  ```bash
  python batch_classify.py \
      --csv-root output/金价 \
      --images-root images/金价 \
      --output-root batch_results \
      --profile mac-cpu \
      --image-proto-root prototypes \
      --keep-image-level
  ```
  会在 `batch_results/` 下生成：
  - `posts/DATE.csv`：每个 `post_id` 一行，包含模态、文本情绪、图像主/细类、置信度、OCR 预览等字段；
  - `images/DATE.csv`（若加 `--keep-image-level`）：保留逐图详情便于调试；
  - `summary.csv`：按日期汇总处理条数、图像数量等元数据。
- `quality_report.py` 用于质量巡检：
  ```bash
  python quality_report.py \
      --posts batch_results/posts/2022-01-01.csv \
      --images batch_results/images/2022-01-01.csv \
      --sample-size 30 \
      --output quality_report.json \
      --sample-output quality_samples.csv
  ```
  - 输出 JSON 统计各模态/情绪/细类分布、置信度低的帖子数量；
  - 可选 `--ground-truth` 传入人工标注（含 `post_id`、`expected_detail_tag` 等）计算准确率与宏 F1；
  - 自动抽样易错样本，生成 `quality_samples.csv` 供人工复核。

先安装依赖：pip install open_clip_torch torch torchvision pandas pillow tqdm（Mac M 系列需对应的 torch 轮子，可选 pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu；若要用 MPS，安装官方 CPU/MPS 版）。
本地 Mac 无 GPU 时推荐：python classify_media.py --profile mac-cpu --date 2022-01-01 --limit 10。若已开启 Apple MPS，则改用 --profile mac-mps。
迁移到服务器后，直接切换命令为：python classify_media.py --profile gpu-server --date 2022-01-01 --output classified_20220101.csv，脚本会自动用 CUDA + ViT-L/14，大幅加速。
下一步建议

在本地以 --dry-run 验证图片/文本匹配是否正确，再逐步增大 --limit。
根据硬件环境装好相应的 torch/open_clip，确认模型下载和推理可行。
服务器跑全量批次后，检查输出 CSV 的分类结果，按需微调提示词或 --text-weight。

python classify_media.py \
    --profile mac-cpu \
    --date 2022-01-01 \
    --image-proto-root prototypes \
    --proto-image-weight 2.0 \
    --proto-text-weight 1.0 \
    --output classified_images.csv
