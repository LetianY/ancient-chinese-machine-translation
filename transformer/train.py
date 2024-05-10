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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
# model.load_state_dict(torch.load('saved/model-7.40905724465847.pt'))
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    # for i, batch in tqdm.tqdm(enumerate(iterator), desc='train', leave=False):
    progress_bar = tqdm.tqdm(enumerate(iterator), desc='train', leave=False, total=len(iterator))
    for i, batch in progress_bar:
        src, trg = batch

        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        # print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    sample_sentence = []
    sample_reference = []
    # output_sentences = []
    # reference_sentences = []
    with torch.no_grad():
        # for i, batch in tqdm.tqdm(enumerate(iterator), desc='eval', leave=False):
        refers = None
        output = None
        progress_bar = tqdm.tqdm(enumerate(iterator), desc='eval', leave=False, total=len(iterator))
        for i, batch in progress_bar:
            output_sentences = []
            reference_sentences = []
            src, trg = batch
            src = src.to(device)
            trg = trg.to(device)

            refers = trg
            output = model(src, trg[:, :-1]) # batch_size, trg_len, trg_vocab_size
            output_reshape = output.contiguous().view(-1, output.shape[-1]) # batch_size * trg_len, trg_vocab_size
            trg = trg[:, 1:].contiguous().view(-1) # batch_size * trg_len

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

        #     output_idxs = output.max(dim=2)[1] # batch_size, trg_len
        #     output_sentences.extend([idx_to_word(output_idxs[i], tgt_vocab).split() for i in range(output_idxs.shape[0])])
        #     reference_sentences.extend([[idx_to_word(refers[i], tgt_vocab).split()] for i in range(refers.shape[0])])

        #     if len(sample_sentence) == 0:
        #         sample_sentence.append(output_sentences[0])
        #         sample_reference.append(reference_sentences[0])

        #     bleu = bleu_score(output_sentences, reference_sentences)
        #     batch_bleu.append(bleu)
        #     progress_bar.set_postfix({'test_loss': '{:.3f}'.format(loss.item()), 'bleu': '{:.3f}'.format(bleu)})

        # avg_bleu = sum(batch_bleu) / len(batch_bleu)
        # print('sample sentence :', sample_sentence[0])
        # print('sample reference :', sample_reference[0])
        print('output sentence :', idx_to_word(output.max(dim=-1)[1][0], tgt_vocab).split())
        print('reference sentence :', idx_to_word(refers[0], tgt_vocab).split())
    # return epoch_loss / len(iterator), avg_bleu
    return epoch_loss / len(iterator)


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    # train_losses, test_losses = [], []
    valid_loss = inf
    # for step in tqdm.tqdm(range(total_epoch), desc='epoch', leave=True):
    progress_bar = tqdm.tqdm(range(total_epoch), desc='epoch', leave=True)
    for step in progress_bar:
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        train_losses.append(train_loss)

        # valid_loss, bleu = evaluate(model, valid_iter, criterion)
        valid_loss = evaluate(model, test_iter, criterion)
        test_losses.append(valid_loss)
        # bleus.append(bleu)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

            # Save the top 5 models in the saved folder
            if len(os.listdir('saved')) > 5:
                file_list = os.listdir('saved')
                file_list = [float(file.split('-')[1].split('.pt')[0]) for file in file_list if '-' in file and '.pt' in file]
                file_list.sort(reverse=True)
                file_list = file_list[:len(file_list) - 5]
                for file in file_list:
                    os.remove('saved/model-{0}.pt'.format(file))

        if os.path.exists('result') is False:
            os.makedirs('result')
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        # f = open('result/bleu.txt', 'w')
        # f.write(str(bleus))
        # f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        # print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        # print(f'\tBLEU Score: {bleu:.3f}')

        progress_bar.set_postfix({'train_loss': '{:.3f}'.format(train_loss),
                                  'test_loss': '{:.3f}'.format(valid_loss),
                                #   'bleu': '{:.3f}'.format(bleu),
                                  'time': '{:2.0f}m {:2.0f}s'.format(epoch_mins, epoch_secs)})


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
