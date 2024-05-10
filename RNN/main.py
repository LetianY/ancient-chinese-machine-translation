import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from terminaltables import AsciiTable

from model_arch.Rnn import EncoderRNN, DecoderRNN, Seq2Seq, sequence_loss
from model_arch.AttModel import AttModel
from bleu import bleu
from data_load import (
    get_batch_indices,
    load_cn_vocab,
    load_en_vocab,
    load_test_data,
    load_train_data,
)
from hyperparameters import Hyperparams as hp
from utils import get_logger
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# log
if not os.path.exists("log"):
    os.mkdir("log")

log_path = os.path.join(
    "log", "log-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".txt"
)
logger = get_logger(log_path)


# Initialize tokenizer and GPT2 model once
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-ancient")
gpt2_model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-ancient")
gpt2_model.eval()  # Only if no training is required for GPT2 model
gpt2_model.to(device)
embedding_length = gpt2_model.transformer.wte.weight.shape[1]


def get_tokenized_id(sentences):
    """Tokenized a batch of sentences."""
    input_ids = [tokenizer.encode(sentence, padding=True, truncation=True, return_tensors='pt').T for sentence in sentences]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    return input_ids

def get_embeddings(input_ids):
    """Generate embeddings for a batch of sentences."""
    with torch.no_grad():
        embeddings = gpt2_model.transformer.wte(input_ids)
    embeddings = embeddings.squeeze()
    return embeddings


# validation script
def bleu_script(f):
    ref_stem = hp.target_data_c_m
    cmd = "{eval_script} {refs} {hyp}".format(
        eval_script=hp.eval_script, refs=ref_stem, hyp=f
    )
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err.decode('utf-8'))
        sys.exit(1)
    bleu = float(out)
    return bleu


def train():
    paras = [["Parameters", "Value"]]
    for key, value in hp.__dict__.items():
        if "__" not in key:
            paras.append([str(key), str(value)])
    paras_table = AsciiTable(paras)
    logger.info("\n" + str(paras_table.table))
    score_list = [
        [
            "epoch_multi_bleu",
            "epoch_bleu_1_gram",
            "epoch_bleu_2_gram",
            "epoch_bleu_3_gram",
            "epoch_bleu_4_gram",
            "epoch",
        ]
    ]

    global_batches = 0
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(cn2idx)
    dec_voc = len(en2idx)
    writer = SummaryWriter()
    # Load data
    X, Y, Sources, Targets = load_train_data()
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    encoder = EncoderRNN(embedding_length, hp.hidden_size).to(device)
    decoder = DecoderRNN(hp.embed_size, hp.hidden_size, dec_voc).to(device)
    model = Seq2Seq(encoder, decoder)
    torch.backends.cudnn.benchmark = True  # may speed up Forward propagation
    if not os.path.exists(hp.model_dir):
        os.makedirs(hp.model_dir)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    for epoch in range(0, hp.num_epochs):
        current_batches = 0
        model.train()
        model.to(device)
        for index, current_index in get_batch_indices(len(Sources), hp.batch_size):
            x_batch = [Sources[i] for i in index]
            # print(x_batch)
            y_batch = [Targets[i] for i in index]
            # y_batch = torch.LongTensor(Targets[index]).to(device)

            input_ids = get_tokenized_id(x_batch)
            # print("input_ids", input_ids.shape)
            input_embeddings = get_embeddings(input_ids).detach()

            # target_tokenized = [tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').T for sentence in y_batch]
            target_ids = get_tokenized_id(y_batch)
            # print("target_id_beforesqeeze", target_ids.shape)
            target_ids = target_ids.squeeze(dim=-1).detach()

            optimizer.zero_grad()
            # print("target_ids", target_ids.shape)
            # print("input_embeddings", input_embeddings.shape)
            if len(input_embeddings.shape) == 2:
                input_embedding = input_embeddings.unsqueeze(0)
            output = model(input_embeddings, target_ids[:, :-1])
            # loss, _, acc = metric(output, y_batch)
            pad_index = cn2idx['<PAD>']
            loss = sequence_loss(output, target_ids, pad_index)
            loss.backward()
            optimizer.step()

            global_batches += 1
            current_batches += 1
            if current_batches % 1 == 0:
                writer.add_scalar(
                    "./loss",
                    scalar_value=loss.detach().cpu().numpy(),
                    global_step=global_batches,
                )
                # writer.add_scalar(
                #     "./acc",
                #     scalar_value=acc.detach().cpu().numpy(),
                #     global_step=global_batches,
                # )

            if (
                current_batches % 10 == 0
                or current_batches == 0
                or current_batches == num_batch
            ):
                logger.info(
                    # "Epoch: {} batch: {}/{}({:.2%}), loss: {:.6}, acc: {:.4}".format(
                    "Epoch: {} batch: {}/{}({:.2%}), loss: {:.6}".format(
                        epoch,
                        current_batches,
                        num_batch,
                        current_batches / num_batch,
                        loss.data.item(),
                        # acc.data.item(),
                    )
                )

        if epoch % hp.check_frequency == 0 or epoch == hp.num_epochs:
            checkpoint_path = hp.model_dir + "/model_epoch_%02d" % epoch + ".pth"
            torch.save(model.state_dict(), checkpoint_path)

        # eval
        score_list = evaluate(model, epoch, writer, score_list)
    writer.close()
    score_table = AsciiTable(score_list)
    logger.info("\n" + score_table.table)


def evaluate(model, epoch, writer, score_list):
    # Load data
    X, Y, Sources, Targets = load_test_data()
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    model.eval()
    model.to(device)
    # Inference
    if not os.path.exists("results"):
        os.mkdir("results")
    list_of_refs = []
    hypotheses = []
    assert hp.batch_size_valid <= len(
        X
    ), "test batch size is large than total data length. Check your data or change batch size."

    for i in range(len(X) // hp.batch_size_valid):
        # Get mini-batches
        # x_batch = X[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        # y_batch = Y[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        sources = Sources[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]
        targets = Targets[i * hp.batch_size_valid : (i + 1) * hp.batch_size_valid]

        # Autoregressive inference
        input_ids = get_tokenized_id(sources)
        input_embeddings = get_embeddings(input_ids)

        # target_tokenized = [tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').T for sentence in y_batch]
        # target_ids = get_tokenized_id(y_batch)

        # start_token = torch.tensor(101, hp.batch_size)
        start_token = torch.tensor([101] * hp.batch_size_valid).unsqueeze(1).to(device)

        # preds_t = torch.LongTensor(
        #     np.zeros((hp.batch_size_valid, hp.maxlen), np.int32)
        # ).to(device)
        # preds = preds_t
        # _, _preds, _ = model(x_, preds)

        # hidden, cell = model.encoder.initHidden(hp.batch_size)
        hidden, cell = model.encoder(input_embeddings)
        
        # Inference loop
        generated_tokens = [start_token]
        for _ in range(hp.maxlen):
            last_token = generated_tokens[-1]  # (batch_size, 1)
            outputs, (hidden, cell) = model.decoder(last_token, hidden, cell)
            _, next_token = outputs.max(dim=2)  # Get the index of the max log-probability
            # print("next_token", next_token.shape)
            # next_token = next_token.squeeze(dim=1)  # (batch_size, 1)
            # print("next_token_after", next_token.shape)
            generated_tokens.append(next_token)
        # generated_tokens = generated_tokens.squeeze(-1)
        generated_tokens = torch.cat(generated_tokens, dim=-1)


        # outputs = model(input_embeddings, start_token[:, :-1])
        # print("outputs:", outputs)
        # _, _preds = torch.max(outputs, -1)
        # preds = _preds.data.cpu().numpy()
        # print("preds:", preds)

        # prepare data for BLEU score
        # print('targets: ', targets)
        # print('preds: ', preds)
        for source, target, pred in zip(sources, targets, generated_tokens):
            # got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
            got = " ".join(idx2en[idx] for idx in pred.cpu().numpy()).strip()
            # print(got)
            # ref = target.split()
            ref = list(target)
            hypothesis = got.split()
            # print(len(ref), len(hypothesis))
            if len(ref) > 3 and len(hypothesis) > 3:
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)
        # if len(list_of_refs) == 0 or len(hypotheses) == 0:
        #     score_list.append([None, None, None, None, None, epoch])
        #     return score_list
            

    ix = np.random.randint(0, hp.batch_size_valid)
    sampling_result = []
    sampling_result.append(["Key", "Value"])
    sampling_result.append(["Source", sources[ix]])
    sampling_result.append(["Target", targets[ix]])
    # sampling_result.append(["Predict", " ".join(idx2en[idx] for idx in preds[ix]).split("</S>")[0].strip()])
    sampling_result.append(["Predict", " ".join(idx2en[idx] for idx in generated_tokens[ix].cpu().numpy()).strip()])
    sampling_table = AsciiTable(sampling_result)
    logger.info("===========sampling START===========")
    logger.info("\n" + str(sampling_table.table))
    logger.info("===========sampling DONE===========")
    # Calculate BLEU score
    hypotheses = [" ".join(x) for x in hypotheses]

    # print(len(list_of_refs), len(hypotheses))
    p_tmp = tempfile.mktemp()
    f_tmp = open(p_tmp, "w")
    f_tmp.write("\n".join(hypotheses))
    f_tmp.close()
    multi_bleu = bleu_script(p_tmp)
    bleu_1_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=1)
    bleu_2_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=2)
    bleu_3_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=3)
    bleu_4_gram = bleu(hypotheses, list_of_refs, smoothing=True, n=4)

    writer.add_scalar("./bleu_1_gram", bleu_1_gram, epoch)
    writer.add_scalar("./bleu_2_gram", bleu_2_gram, epoch)
    writer.add_scalar("./bleu_3_gram", bleu_3_gram, epoch)
    writer.add_scalar("./bleu_4_gram", bleu_4_gram, epoch)
    writer.add_scalar("./multi-bleu", multi_bleu, epoch)

    bleu_result = [
        ["multi-bleu", "bleu_1-gram", "bleu_2-gram", "bleu_3-gram", "bleu_4-gram",],
        [multi_bleu, bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram,],
    ]
    bleu_table = AsciiTable(bleu_result)
    logger.info("BLEU score for Epoch-{}: ".format(epoch) + "\n" + bleu_table.table)
    score_list.append(
        [multi_bleu, bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram, epoch,]
    )

    return score_list


if __name__ == "__main__":
    train()
