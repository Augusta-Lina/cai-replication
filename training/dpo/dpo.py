"""
Direct preference optimisation (Stage 4 of Constitutional AI).

Continues training the SFT LoRA adapter on preference pairs (chosen vs rejected)
from our CAI dataset, mixed with a subsample of UltraFeedback. A frozen copy of
the SFT adapter is the reference model. Settings come from a YAML file:

    uv run training/dpo/dpo.py --config training/dpo/config.yaml
"""

# Block 1: imports
# ---------------------
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import datasets
import torch
import transformers
import trl
from datasets import DatasetDict, concatenate_datasets
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser

logger = logging.getLogger(__name__)

# Block 2: settings for the dataset mixture (from the handbook's configs.py)
# -----------------------
@dataclass
class DatasetConfig:
    """One dataset in the mixture.
    one entry in the list. Five fields, matching the five lines per dataset in config.yaml
    """

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """The whole mixture: a list of datasets plus how to split and shuffle.
    the list of entries plus the seed and the hold-out size."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """trl's script arguments, extended with a `dataset_mixture` key."""

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Datasets to mix, with weights. See config.yaml."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError("dataset_mixture must be a dictionary with a 'datasets' key")

            datasets_list = [
                DatasetConfig(
                    id=d.get("id"),
                    config=d.get("config"),
                    split=d.get("split", "train"),
                    columns=d.get("columns"),
                    weight=d.get("weight", 1.0),
                )
                for d in self.dataset_mixture["datasets"]
            ]

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Every dataset must contribute the same columns, or they can't be concatenated.
            columns_sets = [set(d.columns) for d in datasets_list if d.columns is not None]
            if columns_sets and not all(c == columns_sets[0] for c in columns_sets):
                raise ValueError(f"Column names differ across datasets: {[list(c) for c in columns_sets]}")

# Block 3: load the datasets and mix them (from the handbook's data.py)
# ---------------------
"""
What it does, in order:

1. Simple case: if the config named a single dataset and no mixture, download it and return it as is.
2. Mixture case: for each entry in the list, download the named split from the Hub, keep only the listed columns, shuffle it, and keep the first fraction given by weight. That's the line where UltraChat's 23,000 rows become 69.
3. Combine: stack the pieces into one dataset and shuffle again so our rows and UltraChat's are interleaved.
4. Hold out: if test_split_size is set, cut off that many rows as the evaluation set and return the two parts under the names train and test. Otherwise return everything as train.
"""

def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load one dataset, or build a mixture, according to the script arguments."""
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config)

    if not args.dataset_mixture:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

    logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
    seed = args.dataset_mixture.seed
    datasets_list = []

    for dataset_config in args.dataset_mixture.datasets:
        logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
        ds = datasets.load_dataset(
            dataset_config.id,
            dataset_config.config,
            split=dataset_config.split,
        )
        if dataset_config.columns is not None:
            ds = ds.select_columns(dataset_config.columns)
        if dataset_config.weight is not None:
            ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
            logger.info(
                f"Subsampled dataset '{dataset_config.id}' with weight={dataset_config.weight} to {len(ds)} examples"
            )
        datasets_list.append(ds)

    if not datasets_list:
        raise ValueError("No datasets were loaded from the mixture configuration")

    combined_dataset = concatenate_datasets(datasets_list).shuffle(seed=seed)
    logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

    if args.dataset_mixture.test_split_size is not None:
        combined_dataset = combined_dataset.train_test_split(
            test_size=args.dataset_mixture.test_split_size, seed=seed
        )
        logger.info(f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}")
        return combined_dataset

    return DatasetDict({"train": combined_dataset})
