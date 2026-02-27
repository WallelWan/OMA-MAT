# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OMA-MAT is a machine learning research project (ICLR 2026) implementing **Online Map Association (OMA)** for autonomous driving. It uses a **Map Association Transformer (MAT)** to associate lane-level HD maps with real-time SD map perception. The codebase is built on top of the [Pointcept](https://github.com/Pointcept/Pointcept) framework.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate pointcept-torch2.5.0-cu12.4
export PYTHONPATH=./
```

Requires Python 3.10, PyTorch 2.5.0, CUDA 12.4, and custom C++ ops in `libs/`.

## Common Commands

**Training:**
```bash
sh scripts/train.sh -p python -d oma -c oma-mt-v1m1-l -n oma-mt-v1m1-l
# Or directly:
python tools/train.py --config-file configs/oma/oma-mt-v1m1-l.py --options save_path=exp/oma/oma-mt-v1m1-l
```

**Testing:**
```bash
sh scripts/test.sh -p python -d oma -n oma-mt-v1m1-l -w model_best
# Or directly:
python tools/test.py --config-file configs/oma/oma-mt-v1m1-l.py --options save_path=exp/oma/oma-mt-v1m1-l weight=exp/oma/oma-mt-v1m1-l/model/model_best.pth
```

Script flags: `-g <num_gpus>`, `-m <num_machines>` for distributed training.

**Metric evaluation (two-step, run from `metrics/`):**
```bash
# Step 1: compute per-sample TP/FP/FN into output_dir
python metrics.py \
  --file_dir <pred_json_dir> \
  --output_dir <metric_result_dir> \
  --gt_dir <gt_json_dir> \
  --distance_threshold 1.0

# Step 2: aggregate and print P / R / F1
python read_and_recal_metric.py --file_dir <metric_result_dir>
```

## Architecture

### Configuration System

Configs are **Python files** (not YAML) in `configs/`, using `_base_` inheritance and a registry pattern. Override at runtime with `--options key=value`. The main OMA config is `configs/oma/oma-mt-v1m1-l.py`.

### OMA-Specific Code

The OMA-specific code lives in a few focused files:

- **`pointcept/datasets/oma/mapping.py`** — `MappingDataset`: loads JSON map files containing `link` (roads), `lane`, `vector` (boundaries), and `path` (lane sequences) fields with normalized coordinates.
- **`pointcept/datasets/oma/transform_2d.py`** — 2D augmentation transforms (rotation, flip, jitter, grid sampling, vector processing).
- **`pointcept/models/map_transformer/map_transformer_v1m1.py`** — `MapTransformer`: the core encoder with dual attention — **spatial** (geometric) and **path** (topological lane-to-road). Uses Rotary Positional Encoding (RoPE).
- **`pointcept/models/map_transformer/default.py`** — `DefaultMapping`: wraps the encoder and computes similarity scores between lane and road embeddings.
- **`pointcept/models/map_transformer/loss.py`** — `MappingCELoss` and `MappingCTCLoss` for association supervision.

### Training Infrastructure (from Pointcept)

- **`pointcept/engines/`** — `Trainer` and `Tester` classes, DDP/SLURM launch, config parsing.
- **`pointcept/engines/hooks/`** — Hook-based callbacks for checkpointing, evaluation, logging.
- **`pointcept/utils/registry.py`** — Registry pattern used everywhere for models, datasets, optimizers, schedulers, and hooks. Components are registered with `@MODELS.register_module()` and built via `build_from_cfg`.

### Data Flow

JSON map files → `MappingDataset` → 2D transforms → `Point` batch objects (with offset tracking for variable-size samples) → `MapTransformer` encoder (per-element embeddings) → `DefaultMapping` (pairwise lane-road similarity) → CE + CTC loss.

### Metric Code (`metrics/`)

The evaluation pipeline is separate from the training framework and runs standalone:

- **`make_graph.py`** — shared utility; converts raw JSON data (fields: `sample_point`, `lane`, `link`) into `lane_graph` and `link_graph` dicts used by all metric scripts.
- **`metrics.py`** — main evaluation script. Matches predicted vs. GT sample points, finds all BFS paths between matched point pairs, computes TP/FP/FN per accuracy threshold across **30 path-length bins**. Accepts `--file_dir`, `--output_dir`, `--gt_dir`, `--distance_threshold`.
- **`metrics_frechet.py`** — variant of `metrics.py` that bins paths by **Fréchet distance** (curvature relative to the start-end straight line) instead of path length, using 10 bins. Adds `compute_frechet_distance` / `calculate_frechet_bin` with a path-level cache. Input JSON must have `sample_point_vaild` field (required, not optional).
- **`read_and_recal_metric.py`** — aggregates the per-sample JSON files produced by either metric script and prints P / R / F1 at thresholds 0.5, 0.75, 0.95, and the mean across all thresholds. Accepts `--file_dir`, `--bin_number` (default 30), `--max_bin` (default 30); set both to 10 when using `metrics_frechet.py` output.
- **`compute_frechet_stats.py`** — standalone script for computing Fréchet distance distribution statistics over a dataset.
- **`bak/metrics.py`** — unrelated legacy implementation using a different graph framework (`map.graph`); ignore.

### Key Design Patterns

- The `Point` class (dict-like) is the universal data container passed through the pipeline; keys like `coord`, `feat`, `offset` are accessed by name.
- Batch samples of variable size use **offset tensors** (cumulative element counts) rather than padding.
- The model registry (`MODELS`, `DATASETS`, `HOOKS`, etc.) in `pointcept/utils/registry.py` is how all components are wired together from config dicts.
