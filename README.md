阶段 1：基础资料准备

统计各日 CSV 中文本/图片/图文帖子数量，确定 modality ∈ {text_only, image_only, text_image} 标记方案。
清理并补充图像种子集（prototypes/…），同时从文本中抽样构建手工标注小集合（情绪、主题、风险提示等）。
梳理现有脚本输出，确认所需依赖安装与运行环境（Mac / 服务器）配置。
阶段 2：主类与细分类融合

完善 classify_media.py：
对含图帖子调用图像分类（含原型增强），输出主类与细分类字段。
新增文本处理模块（分词、情绪/主题识别）；缺图时仅跑文本路径。
每条记录统一输出 modality、是否使用文本/图片、图文标签与置信度。
引入 --aggregation post 模式，汇总同一 post_id 的多图结果，并保留 --media-level image 调试选项。
阶段 3：细粒度标注逻辑

technical_chart：识别趋势（上涨/下跌/震荡/不确定），必要时结合OCR读取价格区间。
news_screenshot：OCR 抽取标题与核心段落，做积极/消极/中性判定。
gold_bullion：区分金条/金币/首饰/包装等类型，可先用种子集+CLIP或轻量模型实现。
meme：通过文本 + 图像情绪判断正负面，输出情绪标签。
文本专属补充：对纯文本帖子做相同的情绪/主题分析，确保输出结构一致。
阶段 4：结果汇总与验证

生成新的汇总 CSV（按 post_id），包含日期、作者、文本情绪、图像主类及细分类、modality、置信度、原始文本摘要等。
保留逐图中间结果文件以便调试。
设定质量检验流程：抽样人工复核、计算准确率/F1，并根据问题回滚或调整原型/阈值。
> 已实现：
> - 新增 `batch_classify.py` 批量执行脚本，输出 `posts/DATE.csv`、可选 `images/DATE.csv` 以及整体 `summary.csv`。
> - 新增 `quality_report.py` 巡检工具，支持低置信度样本抽样及与人工标注对比计算准确率/宏 F1。
阶段 5：时间序列特征工程接入

从汇总 CSV 自动提取每日统计特征：各类图片数量、情绪比例、技术图上涨占比等。
将这些特征接入现有 LSTM/预测管线（1GoldPred 项目）并评估对预测性能的提升。
迭代优化，记录实验设置与效果，便于后续上服务器批量运行。
阶段 6：部署与自动化

配置服务器运行脚本：准备依赖环境、同步原型集与配置。
编写命令或脚本实现批量日期处理、日志记录和错误告警。
根据需要拓展到其他关键词或项目（如比特币、景区预测），复用流程。# WeiboData
# WeiboData


## 使用指南

### 环境准备

1. **Conda/Python**：推荐在 `conda` 环境中执行（示例 `WeiboA`）。
   ```bash
   conda activate WeiboA
   pip install open_clip_torch torch torchvision pandas pillow tqdm numpy pytesseract
   ```
2. **Tesseract OCR（含中文语言包）**：
   ```bash
   brew install tesseract tesseract-lang  # macOS 示例
   # 或手动将 chi_sim.traineddata、chi_tra.traineddata 下载至 $(tesseract --print-tessdata-dir)
   ```
   如需指定执行路径，可在 Python 中设置 `pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"`。
3. **目录约定**：保持 `output/<关键词>`（每日一个 CSV）、`images/<关键词>`（按日期存图）、`prototypes/<类别>`（细粒度原型）与脚本路径一致；`.gitignore` 已排除 `batch_results/`、`feature_exports/` 等产物目录。

### classify_media.py：单日/调试运行

```bash
python classify_media.py     --images-root images/金价     --csv-root output/金价     --date 2022-01-01     --profile mac-cpu     --image-proto-root prototypes     --aggregation post     --output classified_posts.csv
```

常用参数：
- `--aggregation {post,image}` 控制帖子级输出或逐图详情；
- `--limit`、`--dry-run` 便于调试；
- `--categories` 使用自定义 prompt；
- `--profile/--device/--model` 控制硬件与 CLIP 版本；
- `--image-proto-root` 指向人工原型目录。

### batch_classify.py：批量产出

```bash
python batch_classify.py     --csv-root output/金价     --images-root images/金价     --output-root batch_results     --profile mac-cpu     --image-proto-root prototypes     --keep-image-level
```

输出：
- `batch_results/posts/DATE.csv`：帖子级结果（模态、细粒度标签、情绪、置信度、OCR 预览等）；
- `batch_results/images/DATE.csv`（若加 `--keep-image-level`）：逐图调试记录；
- `batch_results/summary.csv`：日期层面的处理统计。

### quality_report.py：质量巡检

```bash
python quality_report.py     --posts batch_results/posts/2022-01-01.csv     --images batch_results/images/2022-01-01.csv     --ground-truth ground_truth_template.csv     --sample-size 30     --output quality_report.json     --sample-output quality_samples.csv
```

- 统计模态/情绪/细类分布；
- 支持与人工标注对比计算准确率、宏 F1；
- 输出低置信度样本抽样清单，辅助人工复核。

### extract_features.py：生成时间序列特征

```bash
python extract_features.py     --csv-root output/金价     --images-root images/金价     --profile mac-cpu     --image-proto-root prototypes     --output feature_exports/gold_features_daily.csv
```

产物：
- `feature_exports/gold_features_daily.csv`：按日聚合的特征表，包含帖子量、模态分布、细粒度标签/情绪占比、OCR 统计等；
- `feature_exports/quality_samples.csv`：抽样复核列表。

可将特征 CSV 拷贝或链接到 `/Users/mac/Documents/computerscience/1基于/1GoldPred/data/external/`，供 TFT、LSTM 等模型直接加载。

### 迁移到其他项目/关键词

1. 准备新的 `output/<新关键词>`、`images/<新关键词>` 数据；
2. 在 `prototypes/` 中添加对应类别原型或新的 prompt；
3. 调整脚本参数（如 `--csv-root`、`--images-root`、`--image-proto-root`），即可复用分类与特征流程；
4. 将 `extract_features.py` 输出的日度特征导入目标项目即可用于后续建模；
5. 结合 `ground_truth_template.csv` 与 `quality_report.py` 持续扩充人工标注、监控质量。
