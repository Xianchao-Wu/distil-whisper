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
from typing import Optional, Tuple, Union

from transformers import WhisperConfig, GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
from dataclasses import dataclass

from transformers.modeling_outputs import Seq2SeqLMOutput

@dataclass
class MTPSeq2SeqLMOutput(Seq2SeqLMOutput):
    mtp_logits: torch.FloatTensor = None


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
        "--decoder_mtp_n",
        type=int,
        help="Number of tokens to be predicted for multi-token prediction.",
        default=1, # if 1 then only predict the next 1 token; if > 1 then predict next n tokens.
    )
    parser.add_argument(
        "--decoder_mtp_type",
        type=str,
        help="The type of multi-token prediction, parallel heads or causal heads",
        default="parallel",
        choices=["parallel", "causal"]
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
        self.token_embedding = nn.Embedding(vocab_size, hidden_size) # TODO reuse existing embedding layer
        # student_model.model.decoder.embed_tokens -> Embedding(51866, 1280, padding_idx=50256) 

        # Final projection to vocabulary logits.
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False) # TODO reuse existing head projection
        # student_model.proj_out -> Linear(in_features=1280, out_features=51866, bias=False)

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

def compute_causal_mtp_loss(causal_logits, target_tokens, num_mtp_tokens):
    """
    causal_logits: Tensor of shape (B, T, num_mtp_tokens, vocab_size)
    target_tokens: Tensor of shape (B, T) containing the ground-truth token indices.
    For each time step t (from 0 to T-1), we want to predict tokens at positions t+1, t+2, ..., t+num_mtp_tokens.
    We'll compute loss for positions where such targets exist.
    """
    B, T, num_tokens, vocab_size = causal_logits.size()
    # Only consider time steps where the full target context is available.
    effective_T = T - num_mtp_tokens
    if effective_T <= 0:
        return torch.tensor(0.0, device=causal_logits.device)
    
    loss = 0.0
    # For each time step t in [0, effective_T), the target for the causal head is tokens at positions t+1 ... t+num_mtp_tokens.
    # We can loop over the num_mtp_tokens dimension:
    for i in range(num_mtp_tokens):
        # For position t, target is token at t+i+1.
        target = target_tokens[:, i+1: effective_T + i+1]  # shape: (B, effective_T)
        # Predicted logits at position t, for token i.
        pred_logits = causal_logits[:, :effective_T, i, :]  # shape: (B, effective_T, vocab_size)
        # Flatten and compute cross entropy loss.
        loss += F.cross_entropy(pred_logits.reshape(-1, vocab_size), target.reshape(-1))
    # Average the loss over the num_mtp_tokens predictions.
    loss = loss / num_mtp_tokens
    return loss

def training_step(batch, model, optimizer, kd_loss_fn, alpha_mtp=0.5):
    """
    batch: dictionary containing 'input_features' and 'target_tokens'
           Optionally, include 'teacher_tokens' for the causal head.
    model: an instance of StudentModel.
    kd_loss_fn: function to compute the standard knowledge distillation loss (e.g., between teacher and student logits).
    alpha_mtp: weight for the causal multi-token prediction loss.
    """
    optimizer.zero_grad()
    outputs = model(batch["input_features"], teacher_tokens=batch.get("teacher_tokens"))
    loss_kd = kd_loss_fn(outputs["logits"], batch["target_tokens"])
    mtp_loss = compute_causal_mtp_loss(outputs["causal_mtp_logits"], batch["target_tokens"], model.num_mtp_tokens)
    total_loss = loss_kd + alpha_mtp * mtp_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

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
        self.mtp_proj = nn.Linear(hidden_size, vocab_size * num_tokens, bias=False) # Linear(in_features=1280, out_features=155598, bias=False), to 3 heads in parallel

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        logits = self.mtp_proj(hidden_states)
        # reshape to (batch_size, seq_len, num_tokens, vocab_size)
        batch_size, seq_len, _ = logits.size()
        logits = logits.view(batch_size, seq_len, self.num_tokens, -1)
        return logits

class StudentModelMTPParallel(WhisperForConditionalGeneration):
    #def __init__(self, student_model_whisper, base_config, vocab_size, num_mtp_tokens=3):
    #def __init__(self, config: WhisperConfig, num_mtp_tokens=3):
    def __init__(self, config: WhisperConfig, num_mtp_tokens=2):
        super(StudentModelMTPParallel, self).__init__(config)
        # Suppose self.encoder is the main part of the student (e.g., distilled Whisper encoder)
        #self.encoder = ...  # existing definition
        # A linear projection or decoder head for standard single-token prediction
        #self.standard_head = nn.Linear(base_config.hidden_size, vocab_size)
        # Now add the multi-token prediction head:

        #self.student_model_whisper = self.student_model_whisper

        self.mtp_head = MultiTokenPredictionHeadParallel(
            config.d_model, config.vocab_size, num_tokens=num_mtp_tokens
        )
        
        # TODO obtain num_mtp_tokens and vocab_size from the base_config?
        self.num_mtp_tokens = num_mtp_tokens # TODO put this into the student models' config
        # and also accept command line based modification
        self.vocab_size = config.vocab_size # TODO student_config.vocab_size

    #def forward(self, input_features):
    # TODO this input shall aline with whisper model's forward()'s input parameters!
    def forward(
        self, 
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # Compute hidden states via the existing encoder
        # TODO 从self.student_model_whisper里面，把encoder给解析出来，
        # 并且拿到hidden_states
        #hidden_states = self.student_model_whisper.model.(input_features)  # shape: (B, T, hidden_size)
        #student_outputs = self.student_model_whisper(**batch) # TODO check batch, line 1494 of 
        # 'run_distillation.py' NOTE
        #student_outputs = super().forward(**batch)
        #import ipdb; ipdb.set_trace()
        output_hidden_states = True # force this to be true for mtp's heads
        student_outputs = super().forward(
            input_features = input_features,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            head_mask = head_mask,
            decoder_head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            encoder_outputs = encoder_outputs,
            past_key_values = past_key_values,
            decoder_inputs_embeds = decoder_inputs_embeds,
            decoder_position_ids = decoder_position_ids,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            cache_position = cache_position,
        ) # odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])

        #import ipdb; ipdb.set_trace()
        # Standard single-token logits (if used for distillation loss, etc.)
        # TODO standard_head?
        #logits = self.student_model_whisper.model.standard_head(hidden_states)     # shape: (B, T, vocab_size)
        logits = student_outputs.logits

        #hidden_states = student_outputs.decoder_last_hidden_state # TODO
        hidden_states = student_outputs.decoder_hidden_states[-1] # decoder last hidden state

        # Also compute multi-token logits:
        mtp_logits = self.mtp_head(hidden_states)        # shape: (B, T, num_mtp_tokens, vocab_size)
        # TODO change this and follow Seq2SeqLMOutput
        #return {
        #    "logits": logits, 
        #    "mtp_logits": mtp_logits,
        #    "encoder_last_hidden_state": student_outputs.encoder_last_hidden_state,
        #    "loss": student_outputs.loss
        #}
        return MTPSeq2SeqLMOutput(
            loss=student_outputs.loss,
            logits=logits,
            mtp_logits=mtp_logits,
            encoder_last_hidden_state=student_outputs.encoder_last_hidden_state,
        )
        # torch.Size([1, 1, 51866]) for logits and torch.Size([1, 1, 3, 51866]) for mtp_logits NOTE

import torch.nn.functional as F

def compute_parallel_mtp_loss(mtp_logits, target_tokens, num_mtp_tokens):
    """
    mtp_logits: Tensor of shape (B, T, num_mtp_tokens, vocab_size)
    target_tokens: LongTensor of shape (B, T) containing the ground-truth token indices.
    For causal prediction, at each time step t, we want to predict tokens at positions t+1, ..., t+num_mtp_tokens.
    We thus compute loss only for positions where these targets exist.
    """
    B, T, num_tokens, vocab_size = mtp_logits.size() # 1, 447, 3, 51866
    # Ensure that T is long enough
    if T <= num_mtp_tokens:
        return 0.0

    # We only compute multi-token loss for positions 0..T-num_mtp_tokens
    # For each position t, the targets are tokens at positions t+1 ... t+num_mtp_tokens.
    # Create a target tensor of shape (B, T - num_mtp_tokens, num_mtp_tokens)
    target_seq = []
    for t in range(T - num_mtp_tokens):
        # For position t, slice tokens from t+1 to t+1+num_mtp_tokens
        target_seq.append(target_tokens[:, t+1 : t+1+num_mtp_tokens]) # NOTE [tensor([[-100, -100, -100]], device='cuda:0'), tensor([[-100, -100, -100]], device='cuda:0')], t -> t+1, t+2, t+3, next 3 tokens, duplicated? what I want is t -> t+2, t+3, t+4 since t to t+1 is done in main-model already!
    # Stack along the time dimension to form a tensor of shape (B, T - num_mtp_tokens, num_mtp_tokens)
    target_mtp = torch.stack(target_seq, dim=1) # -> [1, 444, 3]

    # Corresponding predictions: use only the first T - num_mtp_tokens time steps
    pred_mtp = mtp_logits[:, :T - num_mtp_tokens, :, :]  # shape: (B, T - num_mtp_tokens, num_mtp_tokens, vocab_size)

    # Flatten the tensors to compute cross-entropy loss
    loss = F.cross_entropy(
        pred_mtp.reshape(-1, vocab_size),   # (B*(T-num_mtp_tokens)*num_mtp_tokens, vocab_size)
        target_mtp.reshape(-1)                # (B*(T-num_mtp_tokens)*num_mtp_tokens,)
    )
    return loss # t -> t+1, t+2, t+3

def compute_parallel_mtp_loss2(mtp_logits, target_tokens, num_mtp_tokens):
    """
    mtp_logits: Tensor of shape (B, T, num_mtp_tokens, vocab_size)
    target_tokens: LongTensor of shape (B, T) containing the ground-truth token indices.
    For causal prediction, at each time step t, we want to predict tokens at positions t+2, ..., t+1+num_mtp_tokens.
    We thus compute loss only for positions where these targets exist.
    """
    B, T, num_tokens, vocab_size = mtp_logits.size() # 1, 447, 3, 51866
    # Ensure that T is long enough
    if T <= num_mtp_tokens:
        return 0.0

    # We only compute multi-token loss for positions 0..T-num_mtp_tokens
    # For each position t, the targets are tokens at positions t+2 ... t+1+num_mtp_tokens.
    # Create a target tensor of shape (B, T - num_mtp_tokens-1, num_mtp_tokens)
    target_seq = []
    for t in range(T - num_mtp_tokens - 1):
        # For position t, slice tokens from t+2 to t+2+num_mtp_tokens
        target_seq.append(target_tokens[:, t+2 : t+2+num_mtp_tokens]) # NOTE [tensor([[-100, -100, -100]], device='cuda:0'), tensor([[-100, -100, -100]], device='cuda:0')], t -> t+1, t+2, t+3, next 3 tokens, duplicated? what I want is t -> t+2, t+3, t+4 since t to t+1 is done in main-model already!
    # Stack along the time dimension to form a tensor of shape (B, T - num_mtp_tokens, num_mtp_tokens)
    target_mtp = torch.stack(target_seq, dim=1) # -> [1, 443, 3]

    # Corresponding predictions: use only the first T - num_mtp_tokens time steps
    pred_mtp = mtp_logits[:, :T - num_mtp_tokens - 1, :, :]  # shape: (B, T - num_mtp_tokens - 1, num_mtp_tokens, vocab_size)

    # Flatten the tensors to compute cross-entropy loss
    loss = F.cross_entropy(
        pred_mtp.reshape(-1, vocab_size),   # (B*(T-num_mtp_tokens)*num_mtp_tokens, vocab_size)
        target_mtp.reshape(-1)                # (B*(T-num_mtp_tokens)*num_mtp_tokens,)
    )
    return loss # t -> t+2, t+3, t+4

# In your training step:
def training_step(batch, model, optimizer, kd_loss_fn, alpha_mtp=0.5):
    """
    batch: a dict containing 'input_features' and 'target_tokens'
    model: instance of StudentModel
    kd_loss_fn: function to compute the standard KD loss between student and teacher outputs
    alpha_mtp: weighting factor for the multi-token prediction loss.
    """
    optimizer.zero_grad()
    outputs = model(batch["input_features"])
    # Standard logits loss (could be a distillation loss against the teacher or direct CE loss)
    loss_kd = kd_loss_fn(outputs["logits"], batch["target_tokens"])

    # Compute multi-token prediction loss
    mtp_loss = compute_parallel_mtp_loss(outputs["mtp_logits"], batch["target_tokens"], model.num_mtp_tokens)
    # Combine losses (the weight alpha_mtp can be tuned)
    total_loss = loss_kd + alpha_mtp * mtp_loss

    total_loss.backward()
    optimizer.step()
    return total_loss.item()

def init_student_model_from_teacher(
    teacher_checkpoint,
    encoder_layers=None,
    decoder_layers=2,
    decoder_mtp_n=1, # 1 as default
    decoder_mtp_type='paralle', # 'parallel mtp' as default
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

    #teacher_config = 

    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        teacher_checkpoint,
        cache_dir=cache_dir,
        subfolder=subfolder,
        low_cpu_mem_usage=True,
    ) # <class 'transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration'> /usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py NOTE
    # 导入教师模型ckpt
    #import ipdb; ipdb.set_trace() 

    processor = WhisperProcessor.from_pretrained(teacher_checkpoint)
    generation_config = GenerationConfig.from_pretrained(teacher_checkpoint)
    generation_config.forced_decoder_ids = None

    teacher_config = teacher_model.config
    print('-'*20)
    print(teacher_config)

    #import ipdb; ipdb.set_trace()

    teacher_encoder_layers = teacher_config.encoder_layers # 32
    teacher_decoder_layers = teacher_config.decoder_layers # 32

    student_config = copy.deepcopy(teacher_config) # <class 'transformers.models.whisper.configuration_whisper.WhisperConfig'> -> '/usr/local/lib/python3.10/dist-packages/transformers/models/whisper/configuration_whisper.py'
    student_config.update(
        {
            "encoder_layers": encoder_layers if encoder_layers is not None else teacher_encoder_layers,
            "decoder_layers": decoder_layers,
        }
    )

    encoder_mapping = np.linspace(0, teacher_encoder_layers - 1, student_config.encoder_layers, dtype=int) # array from 0 to 31; NOTE if student.en_layers=4, then [0, 10, 20, 31]; if student.en_layers=6, then [0, 6, 12, 18, 24, 31] very interesting
    encoder_mapping[-1] = teacher_encoder_layers - 1 # 31 强制最后一层是teacher model的第31层
    
    encoder_map = {}
    for student_layer, teacher_layer in enumerate(encoder_mapping):
        encoder_map[teacher_layer] = student_layer

    if decoder_layers_numbers is None:
        decoder_mapping = np.linspace(0, teacher_decoder_layers - 1, student_config.decoder_layers, dtype=int)
        decoder_mapping[-1] = teacher_decoder_layers - 1 # NOTE 这里是保证teach model的decoder的最后一层一定在student model里面
    else:
        decoder_mapping = decoder_layers_numbers

    decoder_map = {}
    for student_layer, teacher_layer in enumerate(decoder_mapping):
        decoder_map[teacher_layer] = student_layer

    # init the student params from the teacher model
    #import ipdb; ipdb.set_trace() # 根据学生模型的配置文件，构造学生模型:
    # TODO
    #student_model = WhisperForConditionalGeneration(student_config)
    student_model = StudentModelMTPParallel(student_config, decoder_mtp_n)
    print(student_model)
    # NOTE /usr/local/lib/python3.10/dist-packages/transformers/models/whisper/modeling_whisper.py
    #import ipdb; ipdb.set_trace()
    #student_model_mtp = StudentModelMTP(student_model, student_config, vocab_size=12800, num_mtp_tokens=3)

    #import ipdb; ipdb.set_trace() # 根据教师网络的ckpt state dict，构造学生模型的一些位置的参数的取值:
    missing_keys, unexpected_keys = student_model.load_state_dict(teacher_model.state_dict(), strict=False) # NOTE len(missing_keys)=0, len(unexpected_keys)=720
    if len(missing_keys) > 0:
        # NOTE initialize mtp weight and bias by 'proj_out.weight', (no:) 'proj_out.bias'
        # 3*proj_out.weight -> mtp_proj.weight
        proj_out_weight = teacher_model.state_dict()['proj_out.weight'] # torch.Size([51866, 1280])

        mtp_proj_weight = torch.cat([proj_out_weight.clone() for _ in range(decoder_mtp_n)], dim=0) # torch.Size([155598, 1280])
        #student_model.mtp_head.load_state_dict({'mtp_head.mtp_proj.weight': mtp_proj_weight})
        student_model.mtp_head.mtp_proj.weight = nn.Parameter(mtp_proj_weight) # NOTE

        #raise RuntimeError(
        #    "Error(s) in loading state_dict for StudentModelMTPParallel(WhisperForConditionalGeneration). \n"
        #    f"Missing key(s) in state_dict: {missing_keys}"
        #)
    if decoder_layers == teacher_decoder_layers: # NOTE not in
        decoder_keys = [key for key in unexpected_keys if "model.decoder.layers" in key]
        if len(decoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for StudentModelMTPParallel(WhisperForConditionalGeneration). \n"
                f"Unexpected key(s) in state_dict: {decoder_keys}"
            )
    if encoder_layers == teacher_encoder_layers:
        encoder_keys = [key for key in unexpected_keys if "model.encoder.layers" in key] # NOTE len(encoder_keys)=0
        if len(encoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for StudentModelMTPParallel(WhisperForConditionalGeneration). \n"
                f"Unexpected key(s) in state_dict: {encoder_keys}"
            )
        
    for layer in range(teacher_decoder_layers):
        if layer in decoder_map: # {0: 0, 31: 1}, t.de.layer: s.de.layer; key=teacher.de.layer.idx, value=student.de.layer.idx
            # re-introduce pre-defined layers from the teacher
            student_model.model.decoder.layers[decoder_map[layer]].load_state_dict(
                teacher_model.model.decoder.layers[layer].state_dict()
            ) # NOTE 好处=teacher的原来的weight被重复使用了；坏处=需要两者的layer的形状一样才行...

    if encoder_layers is not None:
        for layer in range(teacher_encoder_layers):
            if layer in encoder_map:
                # re-introduce pre-defined layers from the teacher
                student_model.model.encoder.layers[encoder_map[layer]].load_state_dict(
                    teacher_model.model.encoder.layers[layer].state_dict()
                )

    # remove the teacher params and model
    del teacher_model

    #import ipdb; ipdb.set_trace() # 保存学生模型ckpt到本地的hard space
    # save the converted weights and model
    is_save_student = True #False # NOTE TODO
    if is_save_student and save_dir is not None:
        #import ipdb; ipdb.set_trace() # TODO where is the "save_pretrained" method defined???
        student_model.save_pretrained(save_dir) # > /usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py(2397) save_pretrained() 'class PreTrainedModel'
        # we also need to correctly save the processor and generation config
        processor.save_pretrained(save_dir)
        generation_config.save_pretrained(save_dir)

    # check we can do a forward pass with the saved model - first load the weights and processor
    logger.info("Checking we can load the saved model...")
    #student_model = WhisperForConditionalGeneration.from_pretrained(
    student_model = StudentModelMTPParallel.from_pretrained(
        save_dir, # './distil-large-v3-init' or './distil-large-v3-init-debug' > /usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py(2883) from_pretrained() 
        low_cpu_mem_usage=True,
    ) # TODO here something is wrong -> *** ValueError: Trying to set a tensor of shape torch.Size([103732, 1280]) in "weight" (which has shape torch.Size([155598, 1280])), this looks incorrect.
    processor = WhisperProcessor.from_pretrained(save_dir)

    # define some random inputs
    input_features = processor(np.ones(16000), sampling_rate=16000, return_tensors="pt").input_features
    decoder_start_token_id = student_model.config.decoder_start_token_id # 50258
    decoder_input_ids = torch.ones((input_features.shape[0], 1), dtype=torch.long) * decoder_start_token_id

    #import ipdb; ipdb.set_trace()
    # do a forward pass - outputs will be gibberish for the initialised model so we can't check them
    # but we make can sure the model runs as expected
    logger.info("Checking if we can run the converted model forward...")

    #import ipdb; ipdb.set_trace() # TODO NOTE
    sm_out = student_model(
        input_features, 
        decoder_input_ids=decoder_input_ids,
        output_hidden_states=True
    ) # NOTE TODO check the detail of the output of the student model here!
    print(sm_out)

    '''
    odict_keys(['logits', 'past_key_values', 'encoder_last_hidden_state'])
    ipdb> smout['logits'].shape
    torch.Size([1, 1, 51866])
    ipdb> len(smout['past_key_values'])
    2
    ipdb> smout['past_key_values'][0].shape
    ipdb> len(smout['past_key_values'][0])
    4 : decoder-self-attn-key-cache, decoder-self-attn-value-cache, decoder-cross-attn-key-cache, decoder-cross-attn-value-cache 
    totally 2 decoder layers, each layer is with 4 kvcache tensors NOTE
    ipdb> smout['past_key_values'][0][0]
    tensor([[[[ 0.1925, -0.6320,  0.7088,  ...,  0.2387,  0.2302,  0.5410]],

             [[-0.9089,  0.9011, -0.7601,  ...,  0.0793, -0.4765, -0.5775]],

             [[ 0.9323,  0.5366,  0.7171,  ..., -0.8225,  0.0128,  0.8300]],

             ...,

             [[ 0.0064,  1.0563, -0.2240,  ...,  0.9675,  0.0261,  1.0380]],

             [[-0.9611,  0.9479,  0.8635,  ..., -0.8018, -0.9512,  0.7421]],

             [[ 0.6124,  0.5573,  0.5833,  ...,  0.8053, -0.6511,  0.8537]]]],
           grad_fn=<TransposeBackward0>)
    ipdb> smout['encoder_last_hidden_state'].shape
    torch.Size([1, 1500, 1280])
    '''
    logger.info("Conversion successful!")

    if push_to_hub: # False
        student_model.push_to_hub(save_dir)
        processor.push_to_hub(save_dir)
        generation_config.push_to_hub(save_dir)


if __name__ == "__main__":
    args = parse_args()
    #import ipdb; ipdb.set_trace()
    print(args) # Namespace(teacher_checkpoint='openai/whisper-large-v3', subfolder='', encoder_layers=32, decoder_layers=2, decoder_layers_numbers=None, save_dir='./distil-large-v3-init', push_to_hub=False, cache_dir=None)

    init_student_model_from_teacher(
        teacher_checkpoint=args.teacher_checkpoint,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        decoder_mtp_n=args.decoder_mtp_n,
        decoder_mtp_type=args.decoder_mtp_type,
        decoder_layers_numbers=args.decoder_layers_numbers,
        save_dir=args.save_dir,
        push_to_hub=args.push_to_hub,
        cache_dir=args.cache_dir,
        subfolder=args.subfolder,
    )
