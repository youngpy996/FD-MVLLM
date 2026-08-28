[简体中文](README.md) | [English](README_EN.md)

# FD-MVLLM: Fault Diagnosis Based on Multimodal Vibration Data and Large Language Model for Bearing System

![FD-MVLLM framework](https://github.com/youngpy996/FD-MVLLM/blob/main/assets/FD-MVLLM%20framework.svg)

## Key Features

- Multimodal fusion of vibration time series and CWT time-frequency images.
- Text prompts composed of 13 time-domain and frequency-domain statistical features.
- Patch Reprogramming that maps non-text features into the LLM embedding space.
- Parameter-efficient fine-tuning with LoRA.
- An independent classification head following the fusion of the LLM's final hidden layers.
- Stratified train/validation/test splits, gradient accumulation, and gradient clipping.
- Best-validation checkpoint saving, fault-tolerant training resumption, and persistent training state and logs.
- Natural sorting and one-to-one alignment checks for CSV windows and CWT images.

## Workflow

1. Select a dataset and convert the data to CSV files.
2. Convert the CSV data into CWT images with `pic_gen_pool.py`.
3. Configure the parameters in `run_main.py`.
4. Run the experiment and collect the results.

## CWRU Experimental Results (Reproduction and Improvement)

### V1.4

All CSV files sampled at 12 kHz are combined. The main model architecture remains unchanged, and Codex was used to improve model accuracy. In other words, the model offers stronger generalization across multiple operating conditions.

| Metric | Result |
|---|---:|
| Train / validation / test samples | 14,025 / 1,753 / 1,754 |
| Final validation epoch | 50 |
| Final validation loss | 0.072866 |
| Final validation accuracy | **99.42%** |
| Final validation Macro-F1 | **0.9947** |
| Total training time | 6.60 hours |

![Training curves](assets/training_curves.png)

### V1.3

Uses CSV files sampled at 12 kHz and recorded at 1,750 rpm, representing a single operating condition.

### V1.2

This is the earlier `FD-MVLLM-simple` version available under Tags.

Main experimental parameters:

| Parameter | Value |
|---|---:|
| Sampling rate | 12,000 Hz |
| Window / overlap | 1024 / 512 |
| Patch length / stride | 16 / 8 |
| LLM hidden size / layers | 4096 / 4 |
| LoRA rank / alpha | 32 / 64 |
| Batch size / gradient accumulation | 8 / 2 |
| Task LR / LoRA LR | 2e-4 / 1e-4 |
| Weight decay | 1e-3 |
| Label smoothing | 0.01 |
| Training epochs | 50 |

## Project Structure

```text
.
├── README.md                         # Chinese documentation
├── README_EN.md                      # English documentation
├── requirements.txt
├── run_main.py                       # Original/baseline training entry point
├── run_main_accuracy.py              # Recommended accuracy-enhanced entry point
├── run_main_accuracy_resilient.py    # Lazy loading, automatic recovery, and server training
├── pic_gen_pool.py                   # Original CWT generation script
├── pic_gen_pool_accuracy.py          # Recommended CWT generation script
├── pic_gen_pool_cal.py               # Low-frequency-range CWT experiment script
├── layers/
│   └── StandardNorm.py
├── models/
│   └── Classification.py
├── utils/
│   ├── my_read_data.py
│   └── tools.py
├── scripts/
│   └── plot_training_history.py
├── assets/
│   └── training_curves.png
└── results/
    └── cwru_1024_llama/
        ├── config.json
        ├── summary.json
        └── training_history.csv
```

## Installation

Python 3.10 and an NVIDIA GPU with BF16 support are recommended. Training Llama with LoRA rank 32 requires at least 24 GB of GPU memory; 32 GB is preferable for more reliable execution.

```bash
git clone <YOUR_REPOSITORY_URL>
cd FD-MVLLM-GitHub

conda create -n fdmvllm python=3.10 -y
conda activate fdmvllm
pip install -r requirements.txt
```

If you need a specific CUDA version, first install compatible versions of `torch==2.2.2` and `torchvision==0.17.2` according to the official PyTorch installation instructions, and then install the remaining dependencies.

## Data Preparation

The datasets and pretrained LLM weights are not included in this repository. The recommended directory structure is:

```text
data/
└── CWRU/
    ├── csv/
    │   ├── normal.csv
    │   ├── inner_race.csv
    │   ├── outer_race.csv
    │   └── rolling_element.csv
    └── cwt/
        ├── normal/
        ├── inner_race/
        ├── outer_race/
        └── rolling_element/

pretrained/
└── Llama/
    ├── config.json
    ├── tokenizer.model
    └── ...
```

The current data-loading logic assigns class IDs according to the natural ordering of the CSV files and requires each CSV file to have a CWT image directory with the same base name. For four-class classification, organize the input directory into four class CSV files. If one fault class contains multiple CSV files, explicit label-mapping logic must first be added.

## Generating CWT Images

The accuracy-enhanced script is recommended:

```bash
python pic_gen_pool.py \
  --input_dir data/CWRU/csv \
  --output_root data/CWRU/cwt \
  --sampling_frequency 12000 \
  --window_size 1024 \
  --overlap 512 \
  --min_frequency 1 \
  --max_frequency 512 \
```

The generated files are 128×128 RGBA PNG images that match the model's four-channel CNN input. If the CSV files do not have headers, add `--no_header`. To overwrite existing images, explicitly add `--overwrite`.

## Training

The following command reproduces the recorded CWRU experiment with 1,024-point windows:

```bash
python run_main_accuracy.py \
  --csv_root data/CWRU/csv \
  --image_root data/CWRU/cwt \
  --sampling_rate 12000 \
  --llm_model deepseek \
  --llm_model_root pretrained/Llama \
  --window_size 1024 \
  --overlap 512 \
  --patch_len 16 \
  --stride 8 \
  --d_model 32 \
  --llm_layers 4 \
  --train_epochs 50 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --early_stopping_patience 100 \
  --output_dir results/my_cwru_1024_llama
```

Training produces:

- `config.json`: the effective parameters, sample counts, and normalization statistics.
- `training_history.csv`: per-epoch loss, accuracy, F1 score, and learning rate.
- `summary.json`: the best validation results and final test results.
- `best_trainable_state.pt`: the best trainable parameters.

## Replotting the Training Curves

```bash
python scripts/plot_training_history.py \
  results/cwru_1024_llama/training_history.csv \
  assets/training_curves.png \
  --best-epoch 14
```

## Result Interpretation and Reproducibility Notes

1. The current metrics use a window-level stratified random split with a train/validation/test ratio of 8:1:1.
2. The window overlap is 512 points. Adjacent overlapping windows may be assigned to different splits, so these results are not equivalent to bearing-level, operating-condition-level, or cross-domain generalization results.
3. For more rigorous paper comparisons, split the data by bearing ID or original recording to prevent leakage between adjacent windows.
4. The results depend on the exact local Llama version, CWT generation parameters, CSV file ordering, and random seed.
5. The number and ordering of CWT images must exactly match the CSV windows. The training entry point validates both the counts and the RGBA channels.
6. Model accuracy is strongly affected by CWT image quality, the number of model layers, the loss function, and related settings. CWT image quality is particularly important.
7. FD-MVLLM is a fault-diagnosis model intended for datasets such as JNU, CWRU, and PU. Its applicability to bearing remaining useful life (RUL) datasets has yet to be validated.

The latest concise version is available under FD-MVLLM-v1.4. This also distinguishes it from the Time-LLM documentation.

## Citation

[1] M. Jin, S. Wang, L. Ma, Z. Chu, J.Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan, Q.J.a.e.-p. Wen, "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models," 2023, arXiv:2310.01728. https://doi.org/10.48550/arXiv.2310.01728

[2] Li D, Pang Z, Chen Y, et al. "FD-MVLLM: Fault diagnosis based on multimodal vibration data and large language model for bearing system." *Mechanical Systems and Signal Processing*, 2025, 239: 113226.

## Public Datasets

[3] [JNU Bearing Dataset](https://github.com/ClarkGableWang/JNU-Bearing-Dataset)

[4] [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)

[5] [Paderborn University Bearing Data Center](https://mb.uni-paderborn.de/konstruktions-und-antriebstechnik-kat/forschung/bearing-datacenter/data-sets-and-download)
