# MRT-Mamba

Implementation of **MRT-Mamba: Multi-Representation Adaptation for Time Series Forecasting via Dynamic Prompt and Time-Frequency Fusion**.

The model is named `MRT-Mamba` in the command-line interface and implemented in `models/MRTMamba.py`. It contains:

- **Multi-scale time-frequency fusion:** short-patch and long-patch encoders, gated convolutional adaptation, and an FFT amplitude-spectrum branch, followed by feature concatenation and projection.
- **Hierarchical dynamic prompting:** 12 statistical descriptors condition four meta tokens, six pattern tokens, and six numerical tokens. Pattern prototypes use Top-3 Softmax routing.
- **Frozen pretrained Mamba:** prompt and fused patch embeddings are concatenated and processed by Mamba-130M. Backbone parameters are frozen while gradients propagate to the input modules.
- **Output projection:** the selected hidden features of patch tokens are flattened and projected to the forecasting horizon, followed by inverse normalization.

## Repository structure

```text
.
├── data_provider/          # Dataset splits, windows, and data loaders
├── dataset/                # Place downloaded datasets here
├── layers/                 # Patch embedding, prompts, and normalization
├── models/MRTMamba.py      # Forecasting model
├── scripts/                # Dataset-specific launch scripts
├── utils/                  # Time features, validation, and early stopping
├── run_main.py             # Training and evaluation
└── requirements.txt
```

## Installation

Use Python 3.11. Install PyTorch and the build tools first, then install the dependencies:

```bash
conda create -n mrt-mamba python=3.11
conda activate mrt-mamba
pip install torch==2.9.1 setuptools wheel packaging ninja
pip install --no-build-isolation -r requirements.txt
```

The supplied launch scripts use CUDA with BF16 mixed precision. Install a PyTorch build appropriate for the local CUDA environment. Building the Mamba CUDA kernels requires a compatible CUDA toolkit and C++ compiler.

## Data preparation

Dataset sources, benchmark CSV download links, and the required directory layout are listed in [dataset/README.md](dataset/README.md). Dataset files are not included in this repository.

The model downloads `AI-ModelScope/mamba-130m-hf` through ModelScope when its pretrained weights are not already cached. `MODELSCOPE_CACHE` can specify the cache location. Pretrained weights are not included in this repository.

## Quick start

Run a dataset script from the repository root:

```bash
bash scripts/MRTMamba_ETTh1.sh
```

| Dataset | Script | Forecast horizons | Backbone layers |
| --- | --- | --- | --- |
| ETTh1 | `scripts/MRTMamba_ETTh1.sh` | 96, 192, 336, 720 | 12 |
| ETTh2 | `scripts/MRTMamba_ETTh2.sh` | 96, 192, 336, 720 | 24 |
| ETTm1 | `scripts/MRTMamba_ETTm1.sh` | 96, 192, 336, 720 | 24 |
| ETTm2 | `scripts/MRTMamba_ETTm2.sh` | 96, 192, 336, 720 | 24 |
| ECL | `scripts/MRTMamba_ECL.sh` | 96, 192, 336, 720 | 24 |
| Weather | `scripts/MRTMamba_Weather.sh` | 96, 192, 336, 720 | 24 |
| Traffic | `scripts/MRTMamba_Traffic.sh` | 96, 192, 336, 720 | 24 |

All supplied scripts run horizons 96, 192, 336, and 720 with an input length of 512. The model supports up to all 24 pretrained backbone layers through `--llm_layers`.

For example, the ETTh1 horizon-96 configuration can be run directly:

```bash
accelerate launch --mixed_precision bf16 --num_processes 1 run_main.py \
  --is_training 1 --model MRT-Mamba --model_comment main \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 512 --label_len 48 --pred_len 96 --enc_in 7 \
  --d_model 32 --d_ff 128 --llm_layers 12 \
  --small_patch 8 --small_stride 4 --large_patch 24 --large_stride 4 \
  --batch_size 24 --learning_rate 0.01 --weight_decay 0.01 \
  --train_epochs 100 --dropout 0.2 --lradj type1 --clip_grad 0.5 \
  --result_json ./results/ETTh1_96.json
```

Use `python run_main.py --help` for the available arguments. The launcher controls mixed precision; use BF16 for the supplied model implementation.

## Training and outputs

- Adam updates only trainable parameters using MSE loss. Checkpoint selection and early stopping use validation MSE. Test MSE and MAE are reported for the selected validation epoch.
- The meta-token FiLM computation uses FP32 inside the mixed-precision forward pass. Nonfinite gradient entries are replaced with zero before optional gradient clipping.
- Use `--lradj COS` for cosine annealing or `--lradj type1` for step decay. With `type1`, the effective initial learning rate is `learning_rate / 25`.
- Validation and test loaders use ordered batches and include the final incomplete batch. ETTh1/ETTh2 use multivariate windows; ETTm1/ETTm2, ECL, Weather, and Traffic use individual-channel windows. Choose `--batch_size` according to the sampling scheme.
- Shell scripts write logs under `logs/`. `--result_json` writes the run configuration and metrics. `--save_checkpoint` saves the best checkpoint; it is disabled by default.
- `--save_pred` restores the best-validation weights and exports test predictions, targets, and inputs under `pred_results/`. `--save_forecast_vis` exports the last test window's first variable as CSV under `forecast_vis/`.

## Acknowledgements and license

We acknowledge [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [OFA (GPT4TS)](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) for their publicly available forecasting implementations.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the source license and attribution. Datasets and pretrained weights are distributed separately under their respective terms.
