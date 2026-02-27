<div align="center">

  <h1 style="font-size: 32px; font-weight: bold;"> [ICLR 2026] Online Navigation Refinement: Achieving Lane-Level Guidance by Associating Standard-Definition and Online Perception Maps </h1>

  <a href="https://arxiv.org/abs/2507.07487">
    <img src="https://img.shields.io/badge/ArXiv-OMA-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/datasets/wanjiaxu/OMA">
    <img src="https://img.shields.io/badge/🤗 huggingface-Dataset-blue" alt="dataset">
  </a>
  <a href="https://huggingface.co/wanjiaxu/MAT">
    <img src="https://img.shields.io/badge/🤗 huggingface-Model-green" alt="checkpoint">
  </a>
  <a href="https://www.modelscope.cn/datasets/WallelWan/OMA">
    <img src="https://img.shields.io/badge/ModelScope-Dataset-8A2BE2" alt="dataset">
  </a>
  <a href="https://www.modelscope.cn/models/WallelWan/MAT">
    <img src="https://img.shields.io/badge/ModelScope-Model-8A2BE2" alt="checkpoint">
  </a>
  <a href="https://github.com/WallelWan/OMA-MAT">
    <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="checkpoint">
  </a>
</div>


## Online Map Association Benchmark and Framework

Connecting online mapping with hybrid navigation to enable interpretable autonomous driving.

<p align="center">
  <img src="docs/intro.png"/>
</p>

**Key insights**:
- We introduce Online Map Association (OMA), the first benchmark for hybrid navigation-oriented online map association.

<p align="center">
  <img src="docs/dataset.png"/>
</p>

- We introduce Association P-R, a metric for map association that considers the accuracy and precision of topological alignment.

<p align="center">
  <img src="docs/metric.png"/>
</p>

- We propose a Map Association Transformer (MAT), which utilizes path-aware attention and spatial attention mechanisms to enable understanding of geometric and topological correspondences.

<p align="center">
  <img src="docs/model.png"/>
</p>


## News

- 2026/02/27: 🚀 Dataset and checkpoint are now available on HuggingFace and ModelScope.
- 2026/01/26: 🎉 Congratulations, OMA-MAT is accepted by ICLR 2026. We will open the dataset and checkpoint as soon as possible.
- 2025/07/15: First commit.

## Quick Start

### Prepare Dataset

```bash
# Download the OMA dataset to the data/oma directory using the Huggingface CLI:
huggingface-cli download wanjiaxu/OMA --repo-type dataset --local-dir data/oma
```

### Training

```
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d oma -c oma-mt-v1m1-l -n oma-mt-v1m1-l

# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/oma/oma-mt-v1m1-l.py --options save_path=exp/oma/oma-mt-v1m1-l
```

### Test

```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d oma -n oma-mt-v1m1-l -w model_best

# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/oma/oma-mt-v1m1-l.py --options save_path=exp/oma/oma-mt-v1m1-l weight=exp/oma/oma-mt-v1m1-l/model/model_best.pth
```

To use the pretrained checkpoint from HuggingFace directly:

```bash
# Download the pretrained checkpoint
huggingface-cli download wanjiaxu/MAT --local-dir checkpoints/MAT

# Run evaluation with the downloaded checkpoint
export PYTHONPATH=./
python tools/test.py --config-file configs/oma/oma-mt-v1m1-l.py --options save_path=exp/oma/oma-mt-v1m1-l weight=checkpoints/MAT/model_best.pth
```

### Evaluate with Association P-R Metric

After testing, the model outputs prediction JSON files. Run the Association P-R metric evaluation using the scripts in `metrics/`:

```bash
cd metrics

# Step 1: Compute per-sample TP/FP/FN statistics
python metrics.py \
  --file_dir ../exp/oma/oma-mt-v1m1-l/result \
  --output_dir ../exp/oma/oma-mt-v1m1-l/metric_result \
  --gt_dir ../data/oma/val \
  --distance_threshold 1.0

# Step 2: Aggregate results and print P / R / F1
python read_and_recal_metric.py \
  --file_dir ../exp/oma/oma-mt-v1m1-l/metric_result
```

## Model and Dataset

| Resource | HuggingFace | ModelScope |
|---|---|---|
| Dataset (OMA) | [🤗 wanjiaxu/OMA](https://huggingface.co/datasets/wanjiaxu/OMA) | [WallelWan/OMA](https://www.modelscope.cn/datasets/WallelWan/OMA) |
| Checkpoint (MAT) | [🤗 wanjiaxu/MAT](https://huggingface.co/wanjiaxu/MAT) | [WallelWan/MAT](https://www.modelscope.cn/models/WallelWan/MAT) |

## Licence

This project is released under [MIT licence](./LICENSE).

## Acknowledgment

This project is mainly based on the following projects:
- [Pointcept](https://github.com/Pointcept/Pointcept)

The Readme is inspired by [DeepEyes](https://github.com/Visual-Agent/DeepEyes).

## TODO

- [x] Improved documentation and tutorials
- [ ] Open resource the post-process code.
- [x] Open resource the eval metric code.
- [x] Open resource the dataset in huggingface.
- [x] Open resource the checkpoint in huggingface.

## Citation

```
@article{wan2025driving,
  title={Driving by Hybrid Navigation: An Online HD-SD Map Association Framework and Benchmark for Autonomous Vehicles},
  author={Wan, Jiaxu and Wang, Xu and Xie, Mengwei and Chang, Xinyuan and Liu, Xinran and Pan, Zheng and Xu, Mu and Yuan, Ding},
  journal={arXiv preprint arXiv:2507.07487},
  year={2025}
}
```
