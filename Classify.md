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