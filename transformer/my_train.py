import os
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
import tqdm
from torchtext.data.metrics import bleu_score
from my_tokenizer import inference, model, tokenizer


def run(total_epoch, best_loss):
    for i, batch in tqdm.tqdm(enumerate(test)):
        src_sentence = batch[0]
        tgt_sentence = batch[1]
        trans_sentence = inference(src_sentence)[0]
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write(f'src : {src_sentence}')
            f.write(f'tgt : {tgt_sentence}')
            f.write(f'trans : {trans_sentence}\n\n')

        if i == 100:
            break

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
