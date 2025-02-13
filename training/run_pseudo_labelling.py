#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pseudo-labelling audio data using the Whisper model in preparation for distillation.
"""
# You can also adapt this script for your own pseudo-labelling tasks. Pointers for this are left as comments.

import csv
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import (
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
)
from huggingface_hub import HfFolder, create_repo, get_full_repo_name, snapshot_download, upload_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from soundfile import LibsndfileError
from datasets.arrow_dataset import table_iter
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import ipdb; ipdb.set_trace()

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "processor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to load the model weights. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use in the encoder and decoder attention layers. Can be one of:\n"
                "1. `eager` or `None`: default Transformers attention implementation.\n"
                "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
            )
        },
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated. Use `attn_implementation` instead."},
    )

    def __post_init__(self):
        if self.attn_type is not None and self.attn_implementation is None:
            # set attn_implementation in a backwards compatible way
            if self.attn_type == "flash_attn":
                self.attn_implementation = "sdpa"
            elif self.attn_type == "flash_attn_2":
                self.attn_implementation = "flash_attention_2"
            elif self.attn_type in [None, "eager", "sdpa", "flash_attention_2"]:
                self.attn_implementation = self.attn_type
            else:
                raise ValueError(
                    f"Argument `--attn_type` is deprecated, and set to an invalid option `{self.attn_type}`. You should omit the argument `--attn_type`, and instead set `-attention_implementation` to one of the following:\n"
                    "1. `eager` or `None`: default Transformers attention implementation.\n"
                    "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                    "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
                )
            warnings.warn(
                f"Argument `--attn_type` is deprecated. Use `--attn_implementation` instead. Inferring `--attn_implementation={self.attn_implementation} from argument `--attn_type={self.attn_type}`."
            )
        elif self.attn_type is not None and self.attn_implementation is not None:
            raise ValueError(
                "`--attn_type` and `--attn_implementation` are both specified. Only the argument `--attn_implementation`."
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=500,
        metadata={"help": "The batch size to use for the dataset pre-processing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'."},
    )
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    speaker_id_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the speaker id data. Defaults to None."},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    max_label_length: int = field(
        default=256,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    concatenate_audio: bool = field(
        default=True, #@False, #True,
        metadata={"help": "Whether or not to concatenate the audio samples to `max_duration_in_seconds`."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is"
                " especially useful when data preprocessing errors out in distributed"
                " training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with"
                " `preprocessing_only=True` so that the cached datasets can"
                " consequently be loaded in distributed training"
            )
        },
    )
    dataset_split_name: str = field(
        default="train+validation+test",
        metadata={
            "help": (
                "The name of the data set splits to use (via the datasets library)."
                " Defaults to 'train+validation+test'. Multiple splits can be passed by splitting a"
                " list through the '+' character, e.g. 'train+validation' will"
                " pseudo-label both the 'train' and 'validation' splits sequentially."
            )
        },
    )
    wandb_project: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb project."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use dataset's streaming mode to load and pre-process the data."},
    )
    max_samples_per_split: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples per split to this value if set."},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return the timestamps with the text. This enables the `FlaxWhisperTimestampsLogitsProcessor`."
        },
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual distillation. This argument should be set for multilingual distillation "
                "only. For English speech recognition, it should be left as `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
            "This argument should be set for multilingual distillation only. For English speech recognition, it should be left as `None`."
        },
    )
    decode_token_ids: bool = field(
        default=True,
        metadata={"help": "Deprecated. The predicted token ids should always be decoded to text transcriptions."},
    )
    private_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a private dataset for the pseudo-labelled data."},
    )

    def __post_init__(self):
        if not self.decode_token_ids:
            raise ValueError(
                "The argument `--decode_token_ids` is deprecated. The token ids are now always decoded to "
                "their corresponding text string. This is following a fix to the merges of the Whisper tokenizer"
                "on the Hugging Face Hub: https://huggingface.co/openai/whisper-large-v2/discussions/100. "
                "You should either omit the argument `--decode_token_ids`, or set it to True explicitly."
            )


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The start-of-sequence token id of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    prefix: str = "eval",
):
    """Helper function to log all evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    accelerator.log(log_metrics)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    is_use_wandb = False # TODO
    if accelerator.is_main_process and is_use_wandb:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for split
        prefix = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"{prefix}/all_predictions",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data[:num_lines],
        )

        # log incorrect normalised predictions
        str_data = np.asarray(str_data)
        str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"{prefix}/incorrect_predictions",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data_incorrect[:num_lines],
        )


def main():
    # 1. Parse input arguments NOTE 解析输入的参数论元
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses() # here

    # 2. Initialize the accelerator 初始化加速器 NOTE
    # We will let the accelerator handle device placement for us in this example
    # We simply have to specify the training precision and any trackers being used
    # We'll use the same dtype arguments as our JAX/Flax training script and convert
    # it to accelerate format
    if model_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif model_args.dtype == "bfloat16":
        mixed_precision = "bf16" # here
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200)) # from package 'accelerate'
    # InitProcessGroupKwargs(backend='nccl', init_method=None, timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=[kwargs],
    )

    accelerator.init_trackers(project_name=data_args.wandb_project) # 'distil-whisper'

    # 3. Set-up basic logging NOTE 设置基本的logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning() # here
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Load dataset 导入数据集合 NOTE
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    token = model_args.token if model_args.token is not None else HfFolder().get_token()
    # 'hf_iZLVVVWsMVkXsiwLHztRhcUwB???' used get_token() method, my own hf token information
    data_splits = data_args.dataset_split_name.split("+") # 'train+validation+test'
    for split in data_splits:
        with accelerator.main_process_first():
            raw_datasets[split] = load_dataset(
                data_args.dataset_name, # mozilla-foundation/common_voice_16_1
                data_args.dataset_config_name, # 'ja'
                split=split, # 'train', 'validation', 'test'
                cache_dir=data_args.dataset_cache_dir, # '/workspace/asr/distil-whisper/training/'
                token=token, # my hf's token 
                streaming=data_args.streaming, # False, use offline download mode
                num_proc=data_args.preprocessing_num_workers if not data_args.streaming else None,
            ) # NOTE real 'load_dataset'
    '''
    'train':
    Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 9616
    })
    'validation':
    Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 6094
    })

    'test':
    Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 6094
    })
    '''

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--text_column_name` to the"
            " correct text column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )
    
    # 7. Load pretrained model, tokenizer, and feature extractor NOTE 导入预训练好的模型，tokenizer，以及特征提取器
    import ipdb; ipdb.set_trace()
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path), # 'openai/whisper-large-v3'
        cache_dir=model_args.cache_dir, # '/workspace/asr/distil-whisper/training/'
        revision=model_args.model_revision, # 'main'
        token=token, # my hf's token
    )
    print(config)
    '''
    WhisperConfig {
      "_name_or_path": "openai/whisper-large-v3",
      "activation_dropout": 0.0,
      "activation_function": "gelu",
      "apply_spec_augment": false,
      "architectures": [
        "WhisperForConditionalGeneration"
      ],
      "attention_dropout": 0.0,
      "begin_suppress_tokens": [
        220,
        50257
      ],
      "bos_token_id": 50257,
      "classifier_proj_size": 256,
      "d_model": 1280,
      "decoder_attention_heads": 20,
      "decoder_ffn_dim": 5120,
      "decoder_layerdrop": 0.0,
      "decoder_layers": 32,
      "decoder_start_token_id": 50258,
      "dropout": 0.0,
      "encoder_attention_heads": 20,
      "encoder_ffn_dim": 5120,
      "encoder_layerdrop": 0.0,
      "encoder_layers": 32,
      "eos_token_id": 50257,
      "init_std": 0.02,
      "is_encoder_decoder": true,
      "mask_feature_length": 10,
      "mask_feature_min_masks": 0,
      "mask_feature_prob": 0.0,
      "mask_time_length": 10,
      "mask_time_min_masks": 2,
      "mask_time_prob": 0.05,
      "max_length": 448,
      "max_source_positions": 1500,
      "max_target_positions": 448,
      "median_filter_width": 7,
      "model_type": "whisper",
      "num_hidden_layers": 32,
      "num_mel_bins": 128,
      "pad_token_id": 50256,
      "scale_embedding": false,
      "torch_dtype": "float16",
      "transformers_version": "4.43.2",
      "use_cache": true,
      "use_weighted_layer_sum": false,
      "vocab_size": 51866
    }
    '''

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    print(feature_extractor)
    '''
    loading configuration file preprocessor_config.json from cache at /workspace/asr/distil-whisper/training/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1/preprocessor_config.json
    Feature extractor WhisperFeatureExtractor {
      "chunk_length": 30,
      "feature_extractor_type": "WhisperFeatureExtractor",
      "feature_size": 128,
      "hop_length": 160,
      "n_fft": 400,
      "n_samples": 480000,
      "nb_max_frames": 3000,
      "padding_side": "right",
      "padding_value": 0.0,
      "processor_class": "WhisperProcessor",
      "return_attention_mask": false,
      "sampling_rate": 16000
    }
    '''

    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=token,
    )
    print(tokenizer)
    '''WhisperTokenizerFast(name_or_path='openai/whisper-large-v3', vocab_size=50257, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|startoftranscript|>', '<|en|>', '<|zh|>', '<|de|>', '<|es|>', '<|ru|>', '<|ko|>', '<|fr|>', '<|ja|>', '<|pt|>', '<|tr|>', '<|pl|>', '<|ca|>', '<|nl|>', '<|ar|>', '<|sv|>', '<|it|>', '<|id|>', '<|hi|>', '<|fi|>', '<|vi|>', '<|he|>', '<|uk|>', '<|el|>', '<|ms|>', '<|cs|>', '<|ro|>', '<|da|>', '<|hu|>', '<|ta|>', '<|no|>', '<|th|>', '<|ur|>', '<|hr|>', '<|bg|>', '<|lt|>', '<|la|>', '<|mi|>', '<|ml|>', '<|cy|>', '<|sk|>', '<|te|>', '<|fa|>', '<|lv|>', '<|bn|>', '<|sr|>', '<|az|>', '<|sl|>', '<|kn|>', '<|et|>', '<|mk|>', '<|br|>', '<|eu|>', '<|is|>', '<|hy|>', '<|ne|>', '<|mn|>', '<|bs|>', '<|kk|>', '<|sq|>', '<|sw|>', '<|gl|>', '<|mr|>', '<|pa|>', '<|si|>', '<|km|>', '<|sn|>', '<|yo|>', '<|so|>', '<|af|>', '<|oc|>', '<|ka|>', '<|be|>', '<|tg|>', '<|sd|>', '<|gu|>', '<|am|>', '<|yi|>', '<|lo|>', '<|uz|>', '<|fo|>', '<|ht|>', '<|ps|>', '<|tk|>', '<|nn|>', '<|mt|>', '<|sa|>', '<|lb|>', '<|my|>', '<|bo|>', '<|tl|>', '<|mg|>', '<|as|>', '<|tt|>', '<|haw|>', '<|ln|>', '<|ha|>', '<|ba|>', '<|jw|>', '<|su|>', '<|yue|>', '<|translate|>', '<|transcribe|>', '<|startoflm|>', '<|startofprev|>', '<|nospeech|>', '<|notimestamps|>']}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
            50257: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            50258: AddedToken("<|startoftranscript|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            50259: AddedToken("<|en|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
            50260 to 50358 are "<|yue|>", alike language tags

                    50359: AddedToken("<|translate|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        50360: AddedToken("<|transcribe|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        50361: AddedToken("<|startoflm|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        50362: AddedToken("<|startofprev|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        50363: AddedToken("<|nospeech|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        50364: AddedToken("<|notimestamps|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        
        50365 to 51865: 0.00, 0.02, ..., 29.98, 30.00
        "<|30.00|>"
        
'''

    processor = WhisperProcessor.from_pretrained(
        (model_args.processor_name if model_args.processor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=token,
    )
    print(processor)
    '''
    WhisperProcessor:
    - feature_extractor: WhisperFeatureExtractor { ...
    - tokenizer: WhisperTokenizer(name_or_path='openai/whisper-large-v3', vocab_size=50257 ...
    {
      "processor_class": "WhisperProcessor"
    }
    这个有意思，这是直接定义了一个处理器，然后里面包括了音频的特征提取器，以及文本的tokenizer
    '''
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        subfolder=model_args.subfolder,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation, # 'sdpa' scaled dot product attention
    )
    model.eval()
    print(model)
    '''
    loading weights file model.safetensors from cache at /workspace/asr/distil-whisper/training/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors
Instantiating WhisperForConditionalGeneration model under default dtype torch.bfloat16.
Generate config GenerationConfig {
      "begin_suppress_tokens": [
        220,
        50257
      ],
      "bos_token_id": 50257,
      "decoder_start_token_id": 50258,
      "eos_token_id": 50257,
      "max_length": 448,
      "pad_token_id": 50256
    }

    print(model) -> 
    WhisperForConditionalGeneration(
  (model): WhisperModel(
    (encoder): WhisperEncoder(
      (conv1): Conv1d(128, 1280, kernel_size=(3,), stride=(1,), padding=(1,))
      (conv2): Conv1d(1280, 1280, kernel_size=(3,), stride=(2,), padding=(1,))
      (embed_positions): Embedding(1500, 1280)
      (layers): ModuleList(
        (0-31): 32 x WhisperEncoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=1280, out_features=1280, bias=False)
            (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=1280, out_features=5120, bias=True)
          (fc2): Linear(in_features=5120, out_features=1280, bias=True)
          (final_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): WhisperDecoder(
      (embed_tokens): Embedding(51866, 1280, padding_idx=50256)
      (embed_positions): WhisperPositionalEmbedding(448, 1280)
      (layers): ModuleList(
        (0-31): 32 x WhisperDecoderLayer(
          (self_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=1280, out_features=1280, bias=False)
            (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (activation_fn): GELUActivation()
          (self_attn_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): WhisperSdpaAttention(
            (k_proj): Linear(in_features=1280, out_features=1280, bias=False)
            (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=1280, out_features=5120, bias=True)
          (fc2): Linear(in_features=5120, out_features=1280, bias=True)
          (final_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
    )
  )
  (proj_out): Linear(in_features=1280, out_features=51866, bias=False)
)

'''
    # NOTE TODO whisper架构太复杂了，encoder居然有32层，而且decoder居然也是32层。还是人生第一次看到这么对称的
    # 模型架构...

    if model.config.decoder_start_token_id is None: # <bos> for the decoder, = 50258
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return_timestamps = data_args.return_timestamps # True
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        is_multilingual = True # here
        # We need to set the language and task ids for multilingual checkpoints
        tokenizer.set_prefix_tokens(
            language=data_args.language, task=data_args.task, predict_timestamps=return_timestamps
        ) #  true
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )
    else:
        is_multilingual = False

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, # 'audio'
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate), # 16000
    ) # TODO 给音频信息一些处理, Audio(sampling_rate=16000, mono=True, decode=True, id=None)
    
    import ipdb; ipdb.set_trace()
    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate) # 30.0 * 16000=480000
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    ) # 256, not model.config.max_length=448
    audio_column_name = data_args.audio_column_name # 'audio'
    sampling_rate = feature_extractor.sampling_rate # 16000

    preprocessing_batch_size = data_args.preprocessing_batch_size # 256
    num_workers = data_args.preprocessing_num_workers # 8
    dataloader_num_workers = training_args.dataloader_num_workers # 8

    text_column_name = data_args.text_column_name # 'sentence'
    model_input_name = feature_extractor.model_input_names[0] # 'input_features'
    id_column_name = data_args.id_column_name # 'path'
    speaker_id_column_name = data_args.speaker_id_column_name # None
    normalizer = (
        BasicTextNormalizer()
        if data_args.language is not None # TODO 其他语言并没有一个好的文本标准化的处理程序
        else EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    )

    timestamp_position = 3 if is_multilingual else 1 # 3
    decoder_prev_token_id = tokenizer.convert_tokens_to_ids("<|startofprev|>") # 50362
    decoder_eot_token_id = tokenizer.eos_token_id # 50257

    if data_args.max_samples_per_split is not None: # None, not in
        for split in data_splits:
            raw_datasets[split] = (
                raw_datasets[split].take(data_args.max_samples_per_split)
                if data_args.streaming
                else raw_datasets[split].select(range(data_args.max_samples_per_split))
            )

    if speaker_id_column_name is not None: # None, not in
        raw_datasets = raw_datasets.sort(speaker_id_column_name)

    def concatenate_dataset(batch):
        audio_arrays, texts, speaker_ids = [], [], []
        #import ipdb; ipdb.set_trace()
        # skip corrupted samples
        for row in table_iter(batch.pa_table, batch_size=1):
            row = batch.formatter.format_row(row)
            try:
                sample_audio = row[audio_column_name]['array']
                sample_text = row[text_column_name]
                sample_speaker_id = row[speaker_id_column_name] if speaker_id_column_name else None
            except LibsndfileError:
                logger.warning(f"{row[id_column_name]} is corrupted! Skipping sample.")
                continue
            audio_arrays.append(sample_audio)
            texts.append(sample_text)
            speaker_ids.append(sample_speaker_id)

        # initialize concatenations
        concat_audio = [audio_arrays[0]]
        concat_text = [texts[0]]
        concat_speaker_id = [speaker_ids[0]]
        condition_on_prev = [0]

        for audio_array, text, speaker_id in zip(audio_arrays[1:], texts[1:], speaker_ids[1:]):
            is_same_speaker = speaker_id == concat_speaker_id[-1]
            is_concatenable = len(audio_array) + len(concat_audio[-1]) <= max_input_length 
            if is_same_speaker and is_concatenable:
                # inplace concatenation
                concat_audio[-1] = np.append(concat_audio[-1], audio_array)
                concat_text[-1] = concat_text[-1] + " " + text
            else:
                concat_audio.append(audio_array)
                concat_text.append(text)
                concat_speaker_id.append(speaker_id)
                condition_on_prev.append(1 if is_same_speaker else 0)   

        batch[audio_column_name] = [{"array": array, "sampling_rate": sampling_rate} for array in concat_audio]
        batch[text_column_name] = concat_text
        batch[id_column_name] = concat_speaker_id
        batch["condition_on_prev"] = condition_on_prev

        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys()) # ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant']
    if data_args.concatenate_audio and not data_args.streaming:
        with accelerator.main_process_first(): # NOTE here
            raw_datasets = raw_datasets.map(
                concatenate_dataset,
                batched=True,
                batch_size=preprocessing_batch_size, # 256
                num_proc=num_workers,
                remove_columns=set(raw_datasets_features)
                - {audio_column_name, text_column_name, id_column_name, "condition_on_prev"},
                desc="Concatenating dataset...",
            )
        # 合并了若干短的音频:文字信息，到更长的音频和文本序列.

        raw_datasets = raw_datasets.cast_column(
            audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate)
        )
        pretty_name = data_args.dataset_name.split("/")[-1] # 'common_voice_16_1'
        '''
        ipdb> raw_datasets['test'][0]
        {'path': None, 'audio': {'path': None, 'array': array([-3.05175781e-05,  0.00000000e+00,  0.00000000e+00, ...,
               -3.05175781e-05, -3.05175781e-05, -6.10351562e-05]), 'sampling_rate': 16000}, 'sentence': 'リンゴが食べたい 成長産業の創出 ソリア県の県都はソリアである 祖母は、おおむね機嫌よく、サイコロをころがしている。 すると、ブランコ乗りは突然泣き始めた。すっかり驚いた興行主は飛び上がり、いったいどうしたのか、とたずねた。 埼玉県松伏町', 'condition_on_prev': 0}
        ipdb> raw_datasets['test'][0].keys()
        dict_keys(['path', 'audio', 'sentence', 'condition_on_prev'])
        ipdb> raw_datasets['test'][0]['path']
        ipdb> raw_datasets['test'][0]['audio']
        {'path': None, 'array': array([-3.05175781e-05,  0.00000000e+00,  0.00000000e+00, ...,
               -3.05175781e-05, -3.05175781e-05, -6.10351562e-05]), 'sampling_rate': 16000}
        ipdb> raw_datasets['test'][0]['audio']['array'].shape
        (411840,) < 30.0*16000=480,000
        ipdb> raw_datasets['test'][0]['sentence']
        'リンゴが食べたい 成長産業の創出 ソリア県の県都はソリアである 祖母は、おおむね機嫌よく、サイコロをころがしている。 すると、ブランコ乗りは突然泣き始めた。すっかり驚いた興行主は飛び上がり、いったいどうしたのか、とたずねた。 埼玉県松伏町'
        ipdb> raw_datasets['test'][0]['condition_on_prev']
        0
        ipdb> raw_datasets['test'][1]['condition_on_prev']
        1
        ipdb> raw_datasets['test'][2]['condition_on_prev']
        1
        ipdb> raw_datasets['test'][3]['condition_on_prev']
        1
        ipdb> raw_datasets['test'][10]['condition_on_prev']
        1
        '''

        def postprocess_ids(speaker_ids, indices):
            speaker_ids_formatted = []
            for speaker, idx in zip(speaker_ids, indices):
                formatted_idx = f"{pretty_name}-{speaker}-{idx}" if speaker is not None else f"{pretty_name}-{idx}"
                speaker_ids_formatted.append(formatted_idx)
            return {id_column_name: speaker_ids_formatted}
        
        with accelerator.main_process_first():
            raw_datasets = raw_datasets.map(
                postprocess_ids,
                input_columns=[id_column_name], # 'path'
                with_indices=True,
                desc="Setting sample idxs...",
                batched=True,
                batch_size=preprocessing_batch_size,
                num_proc=num_workers,
            ) # raw_datasets['test'][0]['path'] --> 'common_voice_16_1-0'

    elif data_args.concatenate_audio and data_args.streaming:
        raise ValueError(
            "Streaming mode is not yet compatible with concatenating audios to `max_duration_in_seconds`."
            "Either set `--streaming=False` and download the audios locally, or open an issue on the Distil-Whisper repo to request this feature."
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"]) # NOTE 这里是使用feature_extractor来对输入的原始音频信息，进行feature extraction，信息抽取，fft
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0] # 'input_features'

        # process targets
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys()) # ['path', 'audio', 'sentence', 'condition_on_prev']
    file_ids_dataset = IterableDatasetDict() if data_args.streaming else DatasetDict()
    for split in raw_datasets:
        file_ids_dataset[split] = raw_datasets[split][id_column_name]
    if data_args.streaming:
        with accelerator.main_process_first():
            vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets_features)
    else:
        with accelerator.main_process_first():
            vectorized_datasets = raw_datasets.map(
                prepare_dataset, # NOTE 音频特征提取，以及文字执行tokenizer 
                remove_columns=raw_datasets_features,
                num_proc=num_workers,
                desc="preprocess dataset",
            ) # dict_keys(['input_features', 'labels'])
    # NOTE ['input_features'] is a list, 128 feature elements, each element is with 3000 frames. [128,3000] TODO 
    # TODO 如果是使用了新的数据集合，可以先导入为dataset的格式，audio, sentence具备就好了，然后使用
    # whisper自己有的feature_extractor就好了。

    '''
    ipdb> vectorized_datasets['test'][0]['labels']
        [50258, 50266, 50360, 12376, 4824, 39780, 5142, 50067, 20699, 220, 11336, 15353, 3526, 96, 27119, 2972, 5935, 113, 7781, 220, 42668, 12376, 12817, 2862, 234, 2972, 2862, 234, 7182, 3065, 42668, 12376, 12817, 2474, 24719, 220, 12695, 244, 35744, 3065, 1231, 33888, 33350, 5555, 17543, 38291, 234, 5591, 6134, 1231, 23607, 8040, 18066, 17164, 5998, 38789, 5142, 8822, 22979, 1543, 41068, 4895, 3193, 1231, 28889, 11353, 4824, 18066, 2930, 245, 5095, 3065, 40859, 5823, 6847, 96, 7016, 15476, 11429, 3368, 1543, 2659, 2970, 3703, 5095, 24023, 248, 17679, 44089, 8082, 13557, 3065, 34629, 38761, 5708, 5142, 5095, 1231, 1764, 10102, 1764, 18395, 8533, 2972, 3703, 1231, 3193, 3368, 18216, 5555, 3368, 1543, 220, 161, 15375, 8051, 231, 2862, 234, 36897, 7384, 237, 3526, 118, 50257]
        ipdb> len(vectorized_datasets['test'][0]['labels'])
        125
    vectorized_dataset['test'][0]['input_features'] is a list with 128=feat.size elements, each element is with 3000 frames, where 3000 is the max-frame-length. 是关于输入的音频的
    '''

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    if data_args.streaming and dataloader_num_workers > 0:
        logger.warning(
            "Using multiple dataloader num workers with streaming mode will result in different shards of "
            "data being transcribed in parallel. This is not advised if you want to preserve the order of the "
            "audio-text data."
        )

    # Handle the repository creation
    output_dir = training_args.output_dir # './common_voice_16_1_ja_pseudo_labelled'
    if accelerator.is_main_process: # False
        if training_args.push_to_hub:
            if training_args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(output_dir).absolute().name,
                    token=training_args.hub_token,
                )
            else:
                repo_name = training_args.hub_model_id
            create_repo(repo_name, repo_type="dataset", exist_ok=True, token=training_args.hub_token)
            snapshot_download(repo_id=repo_name, repo_type="dataset", local_dir=output_dir, token=training_args.hub_token)

            # Ensure large txt files can be pushed to the Hub with git-lfs
            with open(os.path.join(output_dir, ".gitattributes"), "r+") as f:
                git_lfs_extensions = f.read()
                if "*.csv" not in git_lfs_extensions:
                    f.write("*.csv filter=lfs diff=lfs merge=lfs -text")

        elif output_dir is not None:
            # this is where we'll save our transcriptions
            os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # 8. Load Metric
    metric = evaluate.load("wer")
    '''
    EvaluationModule(name: "wer", module_type: "metric", features: {'predictions': Value(dtype='string', id='sequence'), 'references': Value(dtype='string', id='sequence')}, usage: """
    Compute WER score of transcribed segments against references.

    Args:
        references: List of references for each speech input.
        predictions: List of transcriptions to score.
        concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.

    Returns:
        (float): the word error rate

    Examples:

        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> wer = evaluate.load("wer")
        >>> wer_score = wer.compute(predictions=predictions, references=references)
        >>> print(wer_score)
        0.5
    """, stored examples: 0)

    '''

    def compute_metrics(preds, labels, file_ids):
        # replace padded labels by the padding token
        import ipdb; ipdb.set_trace()
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=False, decode_with_timestamps=return_timestamps)
        print('pred:', pred_str)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print('ref:', label_str)

        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        file_ids = [file_ids[i] for i in range(len(file_ids)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        print('norm_pred:', norm_pred_str)
        print('norm_label:', norm_label_str)

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return {"wer": wer}, pred_str, label_str, norm_pred_str, norm_label_str, file_ids

    def filter_eot_tokens(preds):
        for idx in range(len(preds)):
            # remove the EOT tokens to get the 'true' token length
            token_ids = [token for token in preds[idx] if token != decoder_eot_token_id]
            token_ids = token_ids + [decoder_eot_token_id]
            preds[idx] = token_ids
        return preds

    # 12. Define Training Schedule
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, # <class 'transformers.models.whisper.processing_whisper.WhisperProcessor'>
        decoder_start_token_id=model.config.decoder_start_token_id,  # <|startoftranscript|>, 50258
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length, # 256
    )

    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.generation_config, "num_beams", 1)
    ) # 1

    gen_kwargs = {
        "max_length": max_label_length, # 256
        "num_beams": num_beams, # 1
        "return_timestamps": return_timestamps, # True
    }
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": data_args.language, # 'ja'
                "task": data_args.task, # 'transcribe'
            }
        ) # {'max_length': 256, 'num_beams': 1, 'return_timestamps': True, 'language': 'ja', 'task': 'transcribe'}
    # remove any preset forced decoder ids since these are deprecated
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # 15. Prepare everything with accelerate
    model = accelerator.prepare(model)

    def eval_step_with_save(split="eval"):
        # ======================== Evaluating ==============================
        import ipdb; ipdb.set_trace()
        eval_preds = []
        eval_labels = []
        eval_ids = []
        pred_str = []
        eval_start = time.time()

        eval_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )
        file_loader = DataLoader(
            file_ids_dataset[split],
            batch_size=per_device_eval_batch_size * accelerator.num_processes,
            num_workers=dataloader_num_workers,
        ) # file_ids_dataset['train'] is a list with 1737 file names, alike 'common_voice_16_1-0'

        eval_loader = accelerator.prepare(eval_loader)
        batches = tqdm(eval_loader, desc=f"Evaluating {split}...", disable=not accelerator.is_local_main_process)

        # make the split name pretty for librispeech etc
        split = split.replace(".", "-").split("/")[-1]
        output_csv = os.path.join(output_dir, f"{split}-transcription.csv") # './common_voice_16_1_ja_pseudo_labelled/train-transcription.csv'
        # batch['input_features'].shape=[64=batch.size, 128=feat.dim, 3000=seq.len], batch['labels'].shape=[64,255], with -100 for padding index NOTE
        for step, (batch, file_ids) in enumerate(zip(batches, file_loader)):
            # Generate predictions and pad to max generated length
            generate_fn = model.module.generate if accelerator.num_processes > 1 else model.generate
            import ipdb; ipdb.set_trace()
            # 真正的调用生成函数，生成target token sequence的过程:
            generated_ids = generate_fn(batch["input_features"].to(dtype=torch_dtype), **gen_kwargs) # NOTE fp32 -> bf16 NOTE NOTE NOTE
            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id) # tensor with shape=[64,169]
            # Gather all predictions and targets
            generated_ids, labels = accelerator.gather_for_metrics((generated_ids, batch["labels"]))
            eval_preds.extend(generated_ids.cpu().numpy())
            eval_labels.extend(labels.cpu().numpy())
            eval_ids.extend(file_ids)

            # TODO added for debug only:
            import ipdb; ipdb.set_trace()
            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str, eval_ids = compute_metrics(
                eval_preds, eval_labels, eval_ids
            )

            if step % training_args.logging_steps == 0 and step >= 0:
                batches.write(f"Saving transcriptions for split {split} step {step}")
                accelerator.wait_for_everyone()
                pred_ids = eval_preds[-(len(eval_preds) - len(pred_str)) :]
                pred_ids = filter_eot_tokens(pred_ids)
                pred_str.extend(
                    tokenizer.batch_decode(
                        pred_ids, skip_special_tokens=False, decode_with_timestamps=return_timestamps
                    )
                )
                csv_data = [[eval_ids[i], pred_str[i]] for i in range(len(eval_preds))]

                with open(output_csv, "w", encoding="UTF8", newline="") as f:
                    writer = csv.writer(f)
                    # write multiple rows
                    writer.writerow(["file_id", "whisper_transcript"])
                    writer.writerows(csv_data)

                if training_args.push_to_hub and accelerator.is_main_process:
                    upload_folder(
                        folder_path=output_dir,
                        repo_id=repo_name,
                        repo_type="dataset",
                        token=training_args.hub_token,
                        commit_message=f"Saving transcriptions for split {split} step {step}.",
                    )

        accelerator.wait_for_everyone()
        eval_time = time.time() - eval_start

        # compute WER metric for eval sets
        wer_desc = ""
        if "validation" in split or "test" in split:
            eval_preds = filter_eot_tokens(eval_preds)
            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str, eval_ids = compute_metrics(
                eval_preds, eval_labels, eval_ids
            )
            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
            # Save metrics + predictions
            log_metric(
                accelerator,
                metrics=wer_metric,
                train_time=eval_time,
                prefix=split,
            )
            log_pred(
                accelerator,
                pred_str,
                label_str,
                norm_pred_str,
                norm_label_str,
                prefix=split,
            )
        else:
            pred_ids = eval_preds[-(len(eval_preds) - len(pred_str)) :]
            pred_ids = filter_eot_tokens(pred_ids)
            pred_str.extend(
                tokenizer.batch_decode(pred_ids, skip_special_tokens=False, decode_with_timestamps=return_timestamps)
            )

        batches.write(f"Saving final transcriptions for split {split}.")
        csv_data = [[eval_ids[i], eval_preds[i]] for i in range(len(eval_preds))]
        with open(output_csv, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerow(["file_id", "whisper_transcript"])
            writer.writerows(csv_data)

        # Print metrics
        logger.info(wer_desc)

        if not data_args.streaming:
            raw_datasets[split] = raw_datasets[split].add_column("whisper_transcript", pred_str)
            raw_datasets[split] = raw_datasets[split].add_column("eval_preds", eval_preds)

            def add_concatenated_text(eval_preds, condition_on_prev):
                concatenated_prev = [None]
                for token_ids, condition in zip(eval_preds[:-1], condition_on_prev[1:]):
                    if condition is False:
                        concatenated_prev.append(None)
                    else:
                        prompt_ids = [token for token in token_ids if token != decoder_eot_token_id]
                        prompt_ids = [decoder_prev_token_id] + prompt_ids[timestamp_position:]
                        concatenated_prev.append(prompt_ids)
                return {"condition_on_prev": concatenated_prev}

            if data_args.concatenate_audio:
                with accelerator.main_process_first():
                    raw_datasets[split] = raw_datasets[split].map(
                        add_concatenated_text,
                        input_columns=["eval_preds", "condition_on_prev"],
                        remove_columns=["eval_preds"],
                        desc="Setting condition on prev...",
                        batched=True,
                        batch_size=preprocessing_batch_size,
                        num_proc=num_workers,
                    )

    logger.info("***** Running Labelling *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(
        f"  Total eval batch size (w. parallel & distributed) = {training_args.per_device_eval_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  Predict labels with timestamps = {return_timestamps}")
    for split in data_splits:
        import ipdb; ipdb.set_trace()
        eval_step_with_save(split=split)
        accelerator.wait_for_everyone()
        if training_args.push_to_hub and accelerator.is_main_process:
            upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                repo_type="dataset",
                token=training_args.hub_token,
                commit_message=f"Saving final transcriptions for split {split.replace('.', '-').split('/')[-1]}",
            )
    if not data_args.streaming and accelerator.is_main_process:
        raw_datasets.save_to_disk(output_dir, num_proc=num_workers)
        if training_args.push_to_hub:
            raw_datasets.push_to_hub(repo_name, token=training_args.hub_token, config_name=data_args.dataset_config_name)
    accelerator.end_training()


if __name__ == "__main__":
    main()
