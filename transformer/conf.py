import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 32
max_len = 128
d_model = 128
n_layers = 6
n_heads = 8
ffn_hidden = 512
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 0
epoch = 200
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
