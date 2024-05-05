import torch
import numpy as np


def get_knn(model_emb, text_emb, dist='cos'):
    if dist == 'cos':
        adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
    elif dist == 'l2':
        adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
            model_emb.size(0), -1, -1)
        adjacency = -torch.norm(adjacency, dim=-1)
    topk_out = torch.topk(adjacency, k=6, dim=0)
    return topk_out.values, topk_out.indices


def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb ** 2).sum(-1).view(-1, 1)  # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices


def denoised_fn_round(args, model, text_emb, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(model_emb, text_emb.to(model_emb.device), dist=dist)
    val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds