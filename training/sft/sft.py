"""
Supervised fine-tuning (Stage 3 of Constitutional AI).

Trains a LoRA adapter on the SFT split of our CAI dataset, mixed with a
subsample of UltraChat. Settings come from a YAML file:

    uv run training/sft/sft.py --config training/sft/config.yaml
"""

# Block 1: imports
# -----------------------
import logging
import os
import sys
from dataclasses import dataclass, field #dataclass is how the config keys become python objects
from typing import Any, Optional

import datasets # loads and mixes the data
import torch
import transformers
import trl
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_peft_config # training library
# SFTTrainer runs the loop, SFTConfig and ModelConfig hold the settings, TrlParser reads the YAML, 
# get_peft_config turns the LoRA keys into a LoRA setup

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

# Block 4: load the tokenizer and the model (from the handbook's model_utils.py)
#  ---------------------
def get_tokenizer(model_args: ModelConfig) -> AutoTokenizer:
    """The tokenizer turns text into token ids. The chat template is set later by the trainer."""
    return AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )


def get_model(model_args: ModelConfig, training_args: SFTConfig) -> AutoModelForCausalLM:
    """Load the base model with the number format and attention path from the config."""
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

# Block 5: main (from the handbook's sft.py)
# ---------------------
def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # Logging: print timestamped progress messages to the terminal
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Resume from an earlier checkpoint if the output folder already has one
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Load data, tokenizer, model
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args)
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # The trainer: wires the model, data and settings together
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Baseline: evaluate the untouched model before any training (not in the original)
    if training_args.do_eval:
        logger.info("*** Evaluate baseline (before training) ***")
        metrics = trainer.evaluate(metric_key_prefix="eval_baseline")
        metrics["eval_baseline_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval_baseline", metrics)
        trainer.save_metrics("eval_baseline", metrics)

    # Train
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save the model (the LoRA adapter) and a model card
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
        "tags": ["constitutional-ai", "cai-replication"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True  # restore the inference cache for later use
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Evaluate on the held-out rows
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push to the Hub
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

# Block 6: entry point
# ---------------------
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


