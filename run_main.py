"""
Accuracy-oriented training entry point for FD-MVLLM.

This file intentionally leaves ``run_main.py`` and all model/data source files
unchanged.  The defaults reproduce the paper's important patching parameters
while fixing the dimensional relationship required by the current
implementation:

    d_model = 2 * patch_len

The script also uses stratified 8:1:1 splits, shuffled training batches,
train-only signal normalization, a logits classification head, configurable
LoRA, label smoothing, gradient clipping, and best-validation checkpointing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from transformers import AutoConfig

from models import Classification
from utils import my_read_data

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")


class LogitsClassificationHead(nn.Module):
    """Stable classification head that returns logits for CrossEntropyLoss."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(self.norm(features)))


def stable_time_features(
    data: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """GPU-native, finite time-domain statistics for the LLM prompt."""

    data = data.float()
    mean_value = data.mean(dim=1)
    centered = data - mean_value.unsqueeze(1)
    variance = centered.square().mean(dim=1)
    std_deviation = variance.clamp_min(1e-12).sqrt()
    max_value = data.max(dim=1).values
    min_value = data.min(dim=1).values
    peak_value = data.abs().max(dim=1).values
    rms_value = data.square().mean(dim=1).clamp_min(1e-12).sqrt()
    kurtosis_value = (
        centered.pow(4).mean(dim=1) / variance.square().clamp_min(1e-12) - 3.0
    )
    skewness_value = (
        centered.pow(3).mean(dim=1)
        / std_deviation.pow(3).clamp_min(1e-12)
    )
    crest_factor = peak_value / rms_value.clamp_min(1e-12)
    return (
        mean_value,
        variance,
        std_deviation,
        max_value,
        min_value,
        peak_value,
        kurtosis_value,
        skewness_value,
        rms_value,
        crest_factor,
    )


def stable_fft_features(
    data: torch.Tensor, sampling_rate: int
) -> Tuple[torch.Tensor, ...]:
    """Positive-frequency FFT statistics with one power denominator per sample."""

    data = data.float()
    sequence_length = data.shape[1]
    spectrum = torch.fft.rfft(data, dim=1)
    frequencies = torch.fft.rfftfreq(
        sequence_length,
        d=1.0 / sampling_rate,
        device=data.device,
    )
    power = spectrum.abs().square() / sequence_length
    total_power = power.sum(dim=1).clamp_min(1e-12)
    peak_frequency = frequencies[power.argmax(dim=1)]
    rms_frequency = (
        (power * frequencies.square()).sum(dim=1) / total_power
    ).clamp_min(0).sqrt()
    center_frequency = (power * frequencies).sum(dim=1) / total_power
    return (
        power,
        total_power,
        peak_frequency,
        rms_frequency,
        center_frequency,
    )


def prompt_number(value: torch.Tensor) -> str:
    return f"{float(value.detach().float().item()):.6g}"


class AccuracyModel(Classification.Model):
    """FD-MVLLM with layer fusion followed by an explicit classification head."""

    def __init__(self, configs: argparse.Namespace) -> None:
        super().__init__(configs)
        # Remove the base class's Softmax head. CrossEntropyLoss must receive
        # logits from an explicit classification head.
        del self.output_projection
        self.classification_head = LogitsClassificationHead(
            configs.llm_dim,
            configs.num_class,
            configs.dropout,
        )
        self.layer_mix_count = min(4, configs.llm_layers)
        self.layer_mix_logits = nn.Parameter(
            torch.zeros(self.layer_mix_count, dtype=torch.float32)
        )
        self.max_prompt_length = configs.max_prompt_length
        self.tokenizer.padding_side = "left"
        self.description = (
            "This dataset contains bearing vibration time series and aligned "
            "CWT time-frequency images for condition classification. "
            f"The vibration sampling frequency is {configs.sampling_rate} Hz."
        )

    def classify(
        self, batch_csv: torch.Tensor, batch_images: torch.Tensor
    ) -> torch.Tensor:
        x_enc = batch_csv
        (
            mean_value,
            variance,
            std_deviation,
            max_value,
            min_value,
            peak_value,
            kurtosis_value,
            skewness_value,
            rms_value,
            crest_factor,
        ) = stable_time_features(x_enc)
        (
            _,
            _,
            peak_frequency,
            rms_frequency,
            center_frequency,
        ) = stable_fft_features(x_enc, self.sampling_rate)

        prompts = []
        for batch_index in range(x_enc.shape[0]):
            prompt = (
                "<|start_prompt|>"
                f"Dataset description: {self.description} "
                "Task description: classify the sequence into predefined "
                "bearing fault categories. Input statistics: "
                f"mean {prompt_number(mean_value[batch_index])}, "
                f"variance {prompt_number(variance[batch_index])}, "
                f"standard deviation "
                f"{prompt_number(std_deviation[batch_index])}, "
                f"maximum {prompt_number(max_value[batch_index])}, "
                f"minimum {prompt_number(min_value[batch_index])}, "
                f"peak {prompt_number(peak_value[batch_index])}, "
                f"kurtosis {prompt_number(kurtosis_value[batch_index])}, "
                f"skewness {prompt_number(skewness_value[batch_index])}, "
                f"RMS {prompt_number(rms_value[batch_index])}, "
                f"crest factor {prompt_number(crest_factor[batch_index])}, "
                f"peak frequency "
                f"{prompt_number(peak_frequency[batch_index])}, "
                f"RMS frequency "
                f"{prompt_number(rms_frequency[batch_index])}, "
                f"center frequency "
                f"{prompt_number(center_frequency[batch_index])}"
                "<|end_prompt|>"
            )
            prompts.append(prompt)

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        prompt_ids = tokenized.input_ids.to(x_enc.device)
        prompt_mask = tokenized.attention_mask.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)

        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)
        time_patches = x_enc.to(torch.bfloat16).unfold(
            1, self.patch_len, self.stride
        )
        image_features, _ = self.pic_enc(batch_images)
        image_patches = image_features.unfold(
            1, self.patch_len, self.stride
        )
        signal_patches = torch.cat([time_patches, image_patches], dim=-1)
        signal_embeddings = self.reprogramming_layer(
            signal_patches,
            source_embeddings,
            source_embeddings,
        )

        llm_inputs = torch.cat([prompt_embeddings, signal_embeddings], dim=1)
        signal_mask = torch.ones(
            (x_enc.shape[0], signal_embeddings.shape[1]),
            dtype=prompt_mask.dtype,
            device=x_enc.device,
        )
        attention_mask = torch.cat([prompt_mask, signal_mask], dim=1)
        outputs = self.llm_model(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        selected_hidden = torch.stack(
            outputs.hidden_states[-self.layer_mix_count :],
            dim=0,
        )
        layer_weights = torch.softmax(
            self.layer_mix_logits,
            dim=0,
        ).to(selected_hidden.dtype)
        fused_hidden = (
            selected_hidden * layer_weights[:, None, None, None]
        ).sum(dim=0)
        signal_hidden = fused_hidden[:, -signal_embeddings.shape[1] :, :]
        pooled_signal = signal_hidden.mean(dim=1)
        return self.classification_head(pooled_signal)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Accuracy-oriented FD-MVLLM training without changing original files"
    )

    # Reproducibility and data.
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=12000)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument(
        "--csv_root",
        type=str,
        default=r"data/CWRU/csv",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=r"data/CWRU/cwt",
    )
    parser.add_argument(
        "--disable_signal_normalization",
        action="store_true",
        help="Disable train-statistics normalization of the time-domain signal.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--legacy_data_loader",
        action="store_true",
        help="Use the repository's original unsorted data loader.",
    )

    # Multimodal model.  The paper reports patch_len=16 and stride=8.
    parser.add_argument("--enc_in", type=int, default=1024)
    parser.add_argument("--enc_out", type=int, default=1024)
    parser.add_argument(
        "--patch_len",
        type=int,
        default=16,
        help="Paper setting; creates local vibration patches.",
    )
    parser.add_argument("--stride", type=int, default=8, help="Paper setting.")
    parser.add_argument(
        "--d_model",
        type=int,
        default=32,
        help="Must equal 2 * patch_len because time/image patches are concatenated.",
    )
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--activation", type=str, default="gelu")

    # LLM and LoRA.
    parser.add_argument(
        "--llm_model_root",
        type=str,
        default=r"pretrained/Llama",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="deepseek",
        choices=("deepseek", "LLAMA", "GPT2", "BERT"),
    )
    parser.add_argument(
        "--llm_dim",
        type=int,
        default=0,
        help="0 automatically reads hidden_size from the local model config.",
    )
    parser.add_argument(
        "--llm_layers",
        type=int,
        default=4,
        help="Eight LLM layers are the accuracy-oriented default for 32 GB VRAM.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        nargs="+",
        default=None,
        help="Optional target module suffixes, e.g. q_proj k_proj v_proj o_proj.",
    )

    # Optimization.
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for multimodal fusion and classification modules.",
    )
    parser.add_argument(
        "--llm_learning_rate",
        type=float,
        default=1e-4,
        help="Lower learning rate for LoRA parameters.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--pct_start", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--early_stopping_delta", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to results_accuracy_<LLM directory name>.",
    )
    return parser


def _config_hidden_size(config: object) -> int:
    candidate_configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        candidate_configs.insert(0, text_config)

    for candidate in candidate_configs:
        for attribute in ("hidden_size", "n_embd", "d_model"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError("Cannot infer hidden_size from the model configuration.")


def resolve_llm_dim(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.llm_model == "deepseek":
        model_root = Path(args.llm_model_root)
        if not model_root.exists():
            parser.error(f"llm_model_root does not exist: {model_root}")
        try:
            model_config = AutoConfig.from_pretrained(
                model_root,
                trust_remote_code=True,
                local_files_only=True,
            )
            detected_dim = _config_hidden_size(model_config)
        except (OSError, ValueError) as error:
            parser.error(f"Unable to read hidden_size from {model_root}: {error}")
    else:
        detected_dim = {
            "LLAMA": 4096,
            "GPT2": 768,
            "BERT": 768,
        }[args.llm_model]

    if args.llm_dim == 0:
        args.llm_dim = detected_dim
    elif args.llm_dim != detected_dim:
        parser.error(
            f"llm_dim={args.llm_dim} does not match the loaded model's "
            f"hidden_size={detected_dim}. Use --llm_dim 0 or {detected_dim}."
        )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.d_model != 2 * args.patch_len:
        parser.error(
            f"d_model must be 2 * patch_len in the current fusion code; "
            f"got d_model={args.d_model}, patch_len={args.patch_len}."
        )
    if args.d_model % args.n_heads != 0:
        parser.error("d_model must be divisible by n_heads.")
    if args.enc_in != args.window_size or args.enc_out != args.window_size:
        parser.error(
            "enc_in and enc_out must both equal window_size so that time and "
            "image branches produce the same number of patches."
        )
    if not 0 <= args.overlap < args.window_size:
        parser.error("overlap must satisfy 0 <= overlap < window_size.")
    if args.patch_len <= 0 or args.stride <= 0:
        parser.error("patch_len and stride must be positive.")
    if args.patch_len > args.window_size:
        parser.error("patch_len cannot exceed window_size.")
    if args.batch_size <= 0 or args.train_epochs <= 0:
        parser.error("batch_size and train_epochs must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("gradient_accumulation_steps must be positive.")
    if args.llm_dim <= 0 or args.llm_layers <= 0:
        parser.error("llm_dim and llm_layers must be positive.")
    if args.max_prompt_length <= 0:
        parser.error("max_prompt_length must be positive.")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("lora_rank and lora_alpha must be positive.")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("lora_dropout must be in [0, 1).")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("label_smoothing must be in [0, 1).")
    if not Path(args.csv_root).is_dir():
        parser.error(f"csv_root does not exist or is not a directory: {args.csv_root}")
    if not Path(args.image_root).is_dir():
        parser.error(f"image_root does not exist or is not a directory: {args.image_root}")
    if not Path(args.llm_model_root).exists() and args.llm_model == "deepseek":
        parser.error(f"llm_model_root does not exist: {args.llm_model_root}")


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def stratified_indices(
    labels: torch.Tensor, seed: int, num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_np = labels.reshape(-1).cpu().numpy()
    classes, counts = np.unique(labels_np, return_counts=True)
    if len(classes) != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes, but labels contain "
            f"{classes.tolist()} with counts {counts.tolist()}."
        )
    if counts.min() < 10:
        raise ValueError(
            "Each class needs at least 10 samples for a stable stratified 8:1:1 split; "
            f"counts are {dict(zip(classes.tolist(), counts.tolist()))}."
        )

    all_indices = np.arange(len(labels_np))
    train_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=labels_np,
    )
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=labels_np[holdout_indices],
    )
    return train_indices, val_indices, test_indices


def natural_sort_key(path: Path) -> List[Tuple[int, object]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


def read_aligned_multimodal_data(
    csv_root: str,
    image_root: str,
    window_size: int,
    overlap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read CSV windows and CWT images in deterministic natural-number order."""

    csv_directory = Path(csv_root)
    image_directory = Path(image_root)
    csv_paths = sorted(
        (
            path
            for path in csv_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        ),
        key=natural_sort_key,
    )
    if not csv_paths:
        raise ValueError(f"No CSV files were found in {csv_directory}.")

    step = window_size - overlap
    window_parts: List[torch.Tensor] = []
    image_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    to_tensor = ToTensor()

    for class_index, csv_path in enumerate(csv_paths):
        signal = (
            pd.read_csv(csv_path, usecols=[0])
            .iloc[:, 0]
            .to_numpy(dtype=np.float32, copy=True)
        )
        if len(signal) < window_size:
            raise ValueError(
                f"{csv_path.name} contains {len(signal)} samples, fewer than "
                f"window_size={window_size}."
            )
        windows = np.stack(
            [
                signal[start : start + window_size]
                for start in range(0, len(signal) - window_size + 1, step)
            ],
            axis=0,
        )

        class_image_directory = image_directory / csv_path.stem
        if not class_image_directory.is_dir():
            raise ValueError(
                f"Missing CWT image directory for {csv_path.name}: "
                f"{class_image_directory}"
            )
        image_paths = sorted(
            (
                path
                for path in class_image_directory.iterdir()
                if path.is_file() and path.suffix.lower() in image_extensions
            ),
            key=natural_sort_key,
        )
        if len(image_paths) != len(windows):
            raise ValueError(
                f"Window/image count mismatch for {csv_path.name}: "
                f"{len(windows)} CSV windows versus {len(image_paths)} images."
            )

        class_images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                class_images.append(to_tensor(image.resize((128, 128))))

        window_parts.append(torch.from_numpy(windows))
        image_parts.append(torch.stack(class_images))
        label_parts.append(
            torch.full((len(windows),), class_index, dtype=torch.long)
        )

    return (
        torch.cat(window_parts, dim=0),
        torch.cat(image_parts, dim=0),
        torch.cat(label_parts, dim=0),
    )


def prepare_data(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, float]]:
    if args.legacy_data_loader:
        data_csv, image_data, labels = my_read_data.read_data(
            args.csv_root,
            args.image_root,
            args.window_size,
            args.overlap,
        )
    else:
        data_csv, image_data, labels = read_aligned_multimodal_data(
            args.csv_root,
            args.image_root,
            args.window_size,
            args.overlap,
        )
    data_csv = data_csv.detach().float()
    image_data = image_data.detach().float()
    labels = labels.detach().long().reshape(-1)

    if not (len(data_csv) == len(image_data) == len(labels)):
        raise ValueError(
            "CSV windows, CWT images, and labels must have identical lengths; "
            f"got {len(data_csv)}, {len(image_data)}, and {len(labels)}."
        )
    if image_data.ndim != 4 or image_data.shape[1] != 4:
        raise ValueError(
            "The current CNN expects RGBA CWT images with shape [N, 4, 128, 128]; "
            f"got {tuple(image_data.shape)}."
        )
    if tuple(image_data.shape[-2:]) != (128, 128):
        raise ValueError(
            f"The current CNN expects 128x128 images; got {tuple(image_data.shape[-2:])}."
        )

    train_indices, val_indices, test_indices = stratified_indices(
        labels, args.seed, args.num_class
    )

    normalization = {"signal_mean": 0.0, "signal_std": 1.0}
    if not args.disable_signal_normalization:
        train_tensor_indices = torch.as_tensor(train_indices, dtype=torch.long)
        train_signal = data_csv.index_select(0, train_tensor_indices)
        signal_mean = train_signal.mean()
        signal_std = train_signal.std(unbiased=False).clamp_min(1e-6)
        data_csv.sub_(signal_mean).div_(signal_std)
        normalization = {
            "signal_mean": float(signal_mean.item()),
            "signal_std": float(signal_std.item()),
        }
        del train_signal

    dataset = TensorDataset(data_csv, image_data, labels)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        common_loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        shuffle=False,
        **common_loader_args,
    )
    normalization.update(
        {
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            "test_samples": int(len(test_indices)),
        }
    )
    return train_loader, val_loader, test_loader, normalization


def default_lora_targets(llm_model: str) -> List[str]:
    if llm_model in {"deepseek", "LLAMA"}:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if llm_model == "GPT2":
        return ["c_attn", "c_proj"]
    return ["query", "key", "value"]


def add_accuracy_lora(
    model: Classification.Model, args: argparse.Namespace
) -> Sequence[str]:
    requested_targets = args.lora_targets or default_lora_targets(args.llm_model)
    module_names = [name for name, _ in model.llm_model.named_modules()]
    available_targets = [
        target
        for target in requested_targets
        if any(name == target or name.endswith(f".{target}") for name in module_names)
    ]
    if not available_targets:
        module_suffixes = sorted({name.rsplit(".", 1)[-1] for name in module_names if name})
        raise ValueError(
            f"None of the LoRA targets {requested_targets} exist in the loaded model. "
            f"Available module suffixes include: {module_suffixes[:80]}"
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=available_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.llm_model = get_peft_model(model.llm_model, lora_config)
    return available_targets


def build_model(args: argparse.Namespace) -> Tuple[nn.Module, Sequence[str]]:
    # The original Model freezes the LLM when llm_lora=False.  We then attach a
    # correctly typed and configurable LoRA adapter here.
    args.llm_lora = False
    model = AccuracyModel(args)
    lora_targets: Sequence[str] = []
    if not args.disable_lora:
        lora_targets = add_accuracy_lora(model, args)
    return model.to(torch.bfloat16), lora_targets


def build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> Tuple[AdamW, List[float], Dict[str, int]]:
    buckets: Dict[Tuple[bool, bool], List[nn.Parameter]] = {}
    parameter_counts = {"task": 0, "lora": 0}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_lora = "lora_" in name
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        buckets.setdefault((is_lora, no_decay), []).append(parameter)
        parameter_counts["lora" if is_lora else "task"] += parameter.numel()

    optimizer_groups = []
    max_lrs: List[float] = []
    for (is_lora, no_decay), parameters in buckets.items():
        group_lr = args.llm_learning_rate if is_lora else args.learning_rate
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": group_lr,
                "weight_decay": 0.0 if no_decay else args.weight_decay,
                "group_name": (
                    ("lora" if is_lora else "task")
                    + ("_no_decay" if no_decay else "_decay")
                ),
            }
        )
        max_lrs.append(group_lr)

    if not optimizer_groups:
        raise RuntimeError("No trainable parameters were found.")
    optimizer = AdamW(optimizer_groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer, max_lrs, parameter_counts


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: OneCycleLR,
    criterion: nn.Module,
    accelerator: Accelerator,
    epoch: int,
    args: argparse.Namespace,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(
        loader,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch:03d}/{args.train_epochs:03d}",
    )
    optimizer.zero_grad(set_to_none=True)

    for step, (batch_csv, batch_images, batch_y) in enumerate(progress, start=1):
        batch_y = batch_y.long().reshape(-1)
        with accelerator.accumulate(model):
            logits = model(batch_csv, batch_images, batch_y)
            loss = criterion(logits.float(), batch_y)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        current_batch = batch_y.numel()
        loss_sum += float(loss.detach().item()) * current_batch
        sample_count += current_batch
        if step % args.log_interval == 0:
            progress.set_postfix(
                loss=f"{loss_sum / max(sample_count, 1):.5f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    totals = torch.tensor(
        [loss_sum, float(sample_count)],
        dtype=torch.float64,
        device=accelerator.device,
    )
    totals = accelerator.reduce(totals, reduction="sum")
    return float((totals[0] / totals[1].clamp_min(1)).item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    label_smoothing: float,
) -> Dict[str, float]:
    model.eval()
    logits_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []

    for batch_csv, batch_images, batch_y in loader:
        batch_y = batch_y.long().reshape(-1)
        logits = model(batch_csv, batch_images, batch_y)
        logits, batch_y = accelerator.gather_for_metrics(
            (logits.detach(), batch_y.detach())
        )
        logits_parts.append(logits.float().cpu())
        label_parts.append(batch_y.cpu())

    all_logits = torch.cat(logits_parts, dim=0)
    all_labels = torch.cat(label_parts, dim=0)
    predictions = all_logits.argmax(dim=1)
    loss = F.cross_entropy(
        all_logits,
        all_labels,
        label_smoothing=label_smoothing,
    )
    accuracy = (predictions == all_labels).float().mean()
    macro_f1 = f1_score(
        all_labels.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0,
    )
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy.item()),
        "f1": float(macro_f1),
    }


def save_trainable_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    checkpoint_path: Path,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    if not accelerator.is_main_process:
        return
    unwrapped_model = accelerator.unwrap_model(model)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in unwrapped_model.named_parameters()
        if parameter.requires_grad
    }
    accelerator.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "trainable_state": trainable_state,
        },
        checkpoint_path,
    )


def load_trainable_checkpoint(
    accelerator: Accelerator, model: nn.Module, checkpoint_path: Path
) -> Dict[str, object]:
    accelerator.wait_for_everyone()
    payload = torch.load(checkpoint_path, map_location="cpu")
    unwrapped_model = accelerator.unwrap_model(model)
    _, unexpected = unwrapped_model.load_state_dict(
        payload["trainable_state"], strict=False
    )
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    accelerator.wait_for_everyone()
    return payload


def write_history(history: Iterable[Dict[str, float]], output_path: Path) -> None:
    rows = list(history)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolve_llm_dim(args, parser)
    validate_args(args, parser)
    set_seed(args.seed)

    output_dir = Path(
        args.output_dir
        or f"results_accuracy_1024_{Path(args.llm_model_root).name or args.llm_model}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    start_time = time.time()
    train_loader, val_loader, test_loader, data_info = prepare_data(args)
    model, lora_targets = build_model(args)
    optimizer, max_lrs, parameter_counts = build_optimizer(model, args)

    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=args.train_epochs,
        steps_per_epoch=updates_per_epoch,
        pct_start=args.pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    (
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scheduler,
    )

    config_record = dict(vars(args))
    config_record.update(
        {
            "lora_targets_used": list(lora_targets),
            "trainable_task_parameters": parameter_counts["task"],
            "trainable_lora_parameters": parameter_counts["lora"],
            **data_info,
        }
    )
    if accelerator.is_main_process:
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config_record, handle, ensure_ascii=False, indent=2)

    accelerator.print(
        "Train/validation/test samples: "
        f"{data_info['train_samples']}/{data_info['val_samples']}/{data_info['test_samples']}"
    )
    accelerator.print(
        f"LLM hidden size/layers: {args.llm_dim}/{args.llm_layers}; "
        f"effective batch size per process: "
        f"{args.batch_size * args.gradient_accumulation_steps}"
    )
    accelerator.print(
        "Data loader: "
        + ("legacy repository order" if args.legacy_data_loader else "natural aligned order")
    )
    accelerator.print(
        "Trainable parameters - task modules: "
        f"{parameter_counts['task']:,}, LoRA: {parameter_counts['lora']:,}"
    )
    accelerator.print(f"LoRA targets: {list(lora_targets)}")

    history: List[Dict[str, float]] = []
    checkpoint_path = output_dir / "best_trainable_state.pt"
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.train_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            accelerator,
            epoch,
            args,
        )
        val_metrics = evaluate(
            model, val_loader, accelerator, args.label_smoothing
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        accelerator.print(
            f"Epoch {epoch:03d} | train loss {train_loss:.6f} | "
            f"val loss {val_metrics['loss']:.6f} | "
            f"val accuracy {val_metrics['accuracy']:.4%} | "
            f"val macro-F1 {val_metrics['f1']:.6f}"
        )

        improved = (
            val_metrics["accuracy"]
            > best_val_accuracy + args.early_stopping_delta
        )
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_trainable_checkpoint(
                accelerator, model, checkpoint_path, epoch, val_metrics
            )
        else:
            epochs_without_improvement += 1

        accelerator.wait_for_everyone()
        if epochs_without_improvement >= args.early_stopping_patience:
            accelerator.print(
                f"Early stopping at epoch {epoch}; best epoch was {best_epoch}."
            )
            break

    checkpoint = load_trainable_checkpoint(accelerator, model, checkpoint_path)
    test_metrics = evaluate(model, test_loader, accelerator, args.label_smoothing)
    elapsed_hours = (time.time() - start_time) / 3600.0

    if accelerator.is_main_process:
        write_history(history, output_dir / "training_history.csv")
        summary = {
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation": checkpoint["metrics"],
            "test": test_metrics,
            "elapsed_hours": elapsed_hours,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    accelerator.print(
        f"Best epoch {checkpoint['epoch']} | "
        f"test accuracy {test_metrics['accuracy']:.4%} | "
        f"test macro-F1 {test_metrics['f1']:.6f} | "
        f"results: {output_dir}"
    )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
