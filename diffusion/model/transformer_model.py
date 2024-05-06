from transformers import AutoConfig, GPT2LMHeadModel, GPT2Config, GPT2Model
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

import torch

import numpy as np
import torch.nn as nn

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
            self,
            input_dims,
            output_dims,
            hidden_t_dim,
            dropout=0,
            config=None,
            config_name='uer/gpt2-chinese-ancient',
            vocab_size=None,
            init_pretrained='gpt2',
            logits_mode=1,
    ):
        super().__init__()

        if config is None:
            # config = AutoConfig.from_pretrained(config_name)
            config = GPT2Config.from_pretrained(config_name)
            config.resid_pdrop = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        # embedding layer: can be adapted from transformer
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # time_stamp embedding layer for diffusion
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        # conversion layer for input_dims --> hidden
        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                               nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        # print('init_pretrained:', init_pretrained)
        # print(config)
        if init_pretrained == 'gpt2':
            print('initializing from pretrained model...')
            print(config)
            temp_model = GPT2LMHeadModel.from_pretrained(config_name, config=config)

            # embedding layer
            self.word_embedding = temp_model.transformer.wte
            with torch.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # annotated out if training embedding is opted
            self.lm_head.weight.requires_grad = False
            self.word_embedding.weight.requires_grad = False

            # transformer layers: prediction for diffusion, only use one layer
            self.input_transformers = temp_model.transformer.h[0]

            # positional embedding layer
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_model.transformer.wpe
            # annotated out if training embedding is opted
            self.position_embeddings.weight.requires_grad = False

            del temp_model

        elif init_pretrained == 'no':
            self.input_transformers = GPT2LMHeadModel(config).transformer.h[0]
            self.register_buffer("position_ids", torch.arange(config.n_positions).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.n_positions, config.hidden_size)

        else:
            assert False, "invalid type of init_pretrained"

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:  # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                        text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                                     hidden_repr.size(1))  # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # input_trans_hidden_states = emb_inputs
        # for layer in self.input_transformers:
        #     input_trans_hidden_states = layer(input_trans_hidden_states)
        input_trans_hidden_states = self.input_transformers(emb_inputs)

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)

        return h
