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
Initialise a student Whisper model from a pre-trained teacher model for
teacher-student distillation.
"""

import argparse
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialise a student Whisper model from a teacher model, copying the relevant layer weights and adjusting the processor as necessary."
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="The HF Hub ID of the teacher checkpoint.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="In case the relevant teacher weights are located inside a subfolder of the model repo on huggingface.co, you "
        "can specify the folder name here.",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=None,
        help="Number of encoder layers to use in the student model. Defaults to all layers from the teacher.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=2,
        help="Number of decoder layers to use in the student model. Defaults to 2 layers.",
    )
    parser.add_argument(
        "--decoder_layers_numbers",
        type=int,
        nargs="*",
        help="Layers numbers of the decoder teacher to use in the student model. Defaults to None, equivalent to taking first and last layer (and equivalent to `--decoder_layers_numbers 0 -1`).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Where to save the student weights and processor.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=bool,
        required=False,
        default=False,
        help="Whether to push the student weights and processor to the Hub.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to store the pretrained models downloaded from huggingface.co",
    )

    args = parser.parse_args()
    return args

class CausalMultiTokenPredictionHead(nn.Module):
    def __init__(
        self, hidden_size, vocab_size, num_tokens=3, num_layers=1, 
        num_heads=4, dropout=0.1, teacher_forcing=True):
        """
        hidden_size: dimension of the encoder hidden states
        vocab_size: vocabulary size
        num_tokens: number of tokens to predict causally per encoder time step
        num_layers, num_heads: parameters for the causal decoder transformer
        dropout: dropout probability
        teacher_forcing: if True, the head expects teacher tokens (for differentiability during training)
        """
        super(CausalMultiTokenPredictionHead, self).__init__()
        self.num_tokens = num_tokens
        self.teacher_forcing = teacher_forcing
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Define a small transformer decoder for autoregressive token generation.
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # A learned start token embedding that begins the autoregressive decoding.
        self.start_token = nn.Parameter(torch.randn(1, hidden_size))
        # Define a token embedding layer (this could be shared with your main model if desired)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # Final projection to vocabulary logits.
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_hidden, teacher_tokens=None):
        """
        encoder_hidden: Tensor of shape (B, T, hidden_size) from the student model encoder.
        teacher_tokens (optional): Tensor of shape (B, T, num_tokens) containing ground-truth token indices
                                   to use for teacher forcing.
        Returns:
          logits: Tensor of shape (B, T, num_tokens, vocab_size) with causal multi-token predictions.
        """
        B, T, H = encoder_hidden.shape
        all_outputs = []

        # Loop over each encoder time step (e.g., for each input token in the encoder output)
        for t in range(T):
            # Use the encoder hidden state at time t as "memory" for the decoder.
            memory = encoder_hidden[:, t, :].unsqueeze(0)  # shape: (1, B, H)
            # Initialize the decoder input with the learned start token (repeated for each example).
            dec_input = self.start_token.unsqueeze(0).expand(1, B, H)  # shape: (1, B, H)
            token_logits = []  # will store logits for each generated token
            # Autoregressively generate num_tokens tokens.
            for i in range(self.num_tokens):
                L = dec_input.size(0)  # current sequence length
                # Create a causal mask (upper-triangular with -inf) of shape (L, L)
                causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=dec_input.device), diagonal=1)
                # Transformer decoder expects target input of shape (target_seq_length, B, H)
                dec_output = self.decoder(dec_input, memory, tgt_mask=causal_mask)
                # Get the representation of the last generated token: shape (B, H)
                last_hidden = dec_output[-1]
                # Project to vocabulary logits: shape (B, vocab_size)
                logits = self.proj(last_hidden)
                token_logits.append(logits.unsqueeze(1))  # append along token dimension
                # Determine the next token to feed into the decoder:
                if self.teacher_forcing and teacher_tokens is not None:
                    # Use teacher forcing: fetch ground truth token for time step t and generation position i.
                    next_token = teacher_tokens[:, t, i]  # shape: (B,)
                else:
                    # Otherwise, use greedy decoding (non-differentiable).
                    next_token = logits.argmax(dim=-1)  # shape: (B,)
                # Get the embedding for the next token.
                next_embed = self.token_embedding(next_token)  # shape: (B, H)
                # Append next_embed to the decoder input sequence (add a time dimension).
                next_embed = next_embed.unsqueeze(0)  # shape: (1, B, H)
                dec_input = torch.cat([dec_input, next_embed], dim=0)  # new shape: (L+1, B, H)
            # Concatenate the logits for this encoder time step: shape (B, num_tokens, vocab_size)
            token_logits = torch.cat(token_logits, dim=1)
            # Append to outputs; we'll stack over time steps.
            all_outputs.append(token_logits.unsqueeze(1))  # shape: (B, 1, num_tokens, vocab_size)
        # Final output shape: (B, T, num_tokens, vocab_size)
        outputs = torch.cat(all_outputs, dim=1)
        return outputs

class StudentModelMTPCausal(nn.Module):
    def __init__(self, base_config, vocab_size, num_mtp_tokens=3):
        super(StudentModelMTPCausal, self).__init__()
        # Existing encoder of the distil-whisper student model.
        self.encoder = ...  # Your existing encoder (e.g., a distilled Whisper encoder)
        # Standard head for single-token prediction (if any)
        self.standard_head = nn.Linear(base_config.hidden_size, vocab_size)
        # Causal multi-token prediction head for autoregressive prediction.
        self.causal_mtp_head = CausalMultiTokenPredictionHead(
            hidden_size=base_config.hidden_size,
            vocab_size=vocab_size,
            num_tokens=num_mtp_tokens,
            num_layers=1,      # adjust as needed
            num_heads=4,       # adjust as needed
            dropout=0.1,
            teacher_forcing=True  # enable teacher forcing during training
        )
        self.num_mtp_tokens = num_mtp_tokens

    def forward(self, input_features, teacher_tokens=None):
        # Get encoder hidden states; shape (B, T, hidden_size)
        hidden_states = self.encoder(input_features)
        # Compute standard logits if required.
        logits = self.standard_head(hidden_states)
        # Compute causal multi-token predictions.
        causal_mtp_logits = self.causal_mtp_head(hidden_states, teacher_tokens=teacher_tokens)
        return {"logits": logits, "causal_mtp_logits": causal_mtp_logits}


class MultiTokenPredictionHeadParallel(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_tokens=3):
        """
        parallel heads for multi-token prediction

        hidden_size: the dimension of the hidden states from the student model
        vocab_size: number of tokens in the vocabulary
        num_tokens: how many tokens to predict at each time step
        """
        super(MultiTokenPredictionHeadParallel, self).__init__()
        self.num_tokens = num_tokens
        # Project from hidden state to (num_tokens * vocab_size) logits.
        self.proj = nn.Linear(hidden_size, vocab_size * num_tokens)

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        logits = self.proj(hidden_states)
        # reshape to (batch_size, seq_len, num_tokens, vocab_size)
        batch_size, seq_len, _ = logits.size()
        logits = logits.view(batch_size, seq_len, self.num_tokens, -1)
        return logits

class StudentModelMTPParallel(nn.Module):
    def __init__(self, student_model_whisper, base_config, vocab_size, num_mtp_tokens=3):
        super(StudentModelMTPParallel, self).__init__()
        # Suppose self.encoder is the main part of the student (e.g., distilled Whisper encoder)
        #self.encoder = ...  # existing definition
        # A linear projection or decoder head for standard single-token prediction
        #self.standard_head = nn.Linear(base_config.hidden_size, vocab_size)
        # Now add the multi-token prediction head:

        self.student_model_whisper = self.student_model_whisper

        self.mtp_head = MultiTokenPredictionHeadParallel(
            base_config.hidden_size, vocab_size, num_tokens=num_mtp_tokens
        )
        
        # TODO obtain num_mtp_tokens and vocab_size from the base_config?
        self.num_mtp_tokens = num_mtp_tokens
        self.vocab_size = vocab_size

    #def forward(self, input_features):
    # TODO this input shall aline with whisper model's forward()'s input parameters!
    def forward(self, batch):
        # Compute hidden states via the existing encoder
        # TODO 从self.student_model_whisper里面，把encoder给解析出来，
        # 并且拿到hidden_states
        #hidden_states = self.student_model_whisper.model.(input_features)  # shape: (B, T, hidden_size)
        student_outputs = self.student_model_whisper(**batch) # TODO check batch, line 1494 of 
        # 'run_distillation.py' NOTE

        # Standard single-token logits (if used for distillation loss, etc.)
        # TODO standard_head?
        #logits = self.student_model_whisper.model.standard_head(hidden_states)     # shape: (B, T, vocab_size)
        logits = student_outputs.logits

        hidden_states = student_outputs.decoder_last_hidden_state # TODO

        # Also compute multi-token logits:
        mtp_logits = self.mtp_head(hidden_states)        # shape: (B, T, num_mtp_tokens, vocab_size)
        return {"logits": logits, "mtp_logits": mtp_logits}

def init_student_model_from_teacher(
    teacher_checkpoint,
    encoder_layers=None,
    decoder_layers=2,
    decoder_layers_numbers=None,
    save_dir=None,
    push_to_hub=None,
    cache_dir=None,
    subfolder="",
):
    if decoder_layers_numbers is not None and len(decoder_layers_numbers) != decoder_layers:
        raise ValueError(
            f"Got {len(decoder_layers_numbers)} layers number for {decoder_layers} decoder layers."
        )

    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        teacher_checkpoint,
        cache_dir=cache_dir,
        subfolder=subfolder,
        low_cpu_mem_usage=True,
    )
    # 导入教师模型ckpt
    import ipdb; ipdb.set_trace() 

    processor = WhisperProcessor.from_pretrained(teacher_checkpoint)
    generation_config = GenerationConfig.from_pretrained(teacher_checkpoint)
    generation_config.forced_decoder_ids = None

    teacher_config = teacher_model.config
    teacher_encoder_layers = teacher_config.encoder_layers
    teacher_decoder_layers = teacher_config.decoder_layers

    student_config = copy.deepcopy(teacher_config)
    student_config.update(
        {
            "encoder_layers": encoder_layers if encoder_layers is not None else teacher_encoder_layers,
            "decoder_layers": decoder_layers,
        }
    )

    encoder_mapping = np.linspace(0, teacher_encoder_layers - 1, student_config.encoder_layers, dtype=int)
    encoder_mapping[-1] = teacher_encoder_layers - 1
    
    encoder_map = {}
    for student_layer, teacher_layer in enumerate(encoder_mapping):
        encoder_map[teacher_layer] = student_layer

    if decoder_layers_numbers is None:
        decoder_mapping = np.linspace(0, teacher_decoder_layers - 1, student_config.decoder_layers, dtype=int)
        decoder_mapping[-1] = teacher_decoder_layers - 1
    else:
        decoder_mapping = decoder_layers_numbers

    decoder_map = {}
    for student_layer, teacher_layer in enumerate(decoder_mapping):
        decoder_map[teacher_layer] = student_layer

    # init the student params from the teacher model
    import ipdb; ipdb.set_trace() # 根据学生模型的配置文件，构造学生模型:
    # TODO
    student_model = WhisperForConditionalGeneration(student_config)

    import ipdb; ipdb.set_trace()
    student_model_mtp = StudentModelMTP(student_model, student_config, vocab_size=12800, num_mtp_tokens=3)

    import ipdb; ipdb.set_trace() # 根据教师网络的ckpt state dict，构造学生模型的一些位置的参数的取值:
    missing_keys, unexpected_keys = student_model.load_state_dict(teacher_model.state_dict(), strict=False)
    if len(missing_keys) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for WhisperForConditionalGeneration. \n"
            f"Missing key(s) in state_dict: {missing_keys}"
        )
    if decoder_layers == teacher_decoder_layers:
        decoder_keys = [key for key in unexpected_keys if "model.decoder.layers" in key]
        if len(decoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for WhisperForConditionalGeneration. \n"
                f"Unexpected key(s) in state_dict: {decoder_keys}"
            )
    if encoder_layers == teacher_encoder_layers:
        encoder_keys = [key for key in unexpected_keys if "model.encoder.layers" in key]
        if len(encoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for WhisperForConditionalGeneration. \n"
                f"Unexpected key(s) in state_dict: {encoder_keys}"
            )
        
    for layer in range(teacher_decoder_layers):
        if layer in decoder_map:
            # re-introduce pre-defined layers from the teacher
            student_model.model.decoder.layers[decoder_map[layer]].load_state_dict(
                teacher_model.model.decoder.layers[layer].state_dict()
            )

    if encoder_layers is not None:
        for layer in range(teacher_encoder_layers):
            if layer in encoder_map:
                # re-introduce pre-defined layers from the teacher
                student_model.model.encoder.layers[encoder_map[layer]].load_state_dict(
                    teacher_model.model.encoder.layers[layer].state_dict()
                )

    # remove the teacher params and model
    del teacher_model

    import ipdb; ipdb.set_trace() # 保存学生模型ckpt到本地的hard space
    # save the converted weights and model
    if save_dir is not None:
        student_model.save_pretrained(save_dir)
        # we also need to correctly save the processor and generation config
        processor.save_pretrained(save_dir)
        generation_config.save_pretrained(save_dir)

    # check we can do a forward pass with the saved model - first load the weights and processor
    logger.info("Checking we can load the saved model...")
    student_model = WhisperForConditionalGeneration.from_pretrained(
        save_dir,
        low_cpu_mem_usage=True,
    )
    processor = WhisperProcessor.from_pretrained(save_dir)

    # define some random inputs
    input_features = processor(np.ones(16000), sampling_rate=16000, return_tensors="pt").input_features
    decoder_start_token_id = student_model.config.decoder_start_token_id
    decoder_input_ids = torch.ones((input_features.shape[0], 1), dtype=torch.long) * decoder_start_token_id

    import ipdb; ipdb.set_trace()
    # do a forward pass - outputs will be gibberish for the initialised model so we can't check them
    # but we make can sure the model runs as expected
    logger.info("Checking if we can run the converted model forward...")
    _ = student_model(input_features, decoder_input_ids=decoder_input_ids).logits
    logger.info("Conversion successful!")

    if push_to_hub:
        student_model.push_to_hub(save_dir)
        processor.push_to_hub(save_dir)
        generation_config.push_to_hub(save_dir)


if __name__ == "__main__":
    args = parse_args()
    import ipdb; ipdb.set_trace()
    print(args)

    init_student_model_from_teacher(
        teacher_checkpoint=args.teacher_checkpoint,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        decoder_layers_numbers=args.decoder_layers_numbers,
        save_dir=args.save_dir,
        push_to_hub=args.push_to_hub,
        cache_dir=args.cache_dir,
        subfolder=args.subfolder,
    )
