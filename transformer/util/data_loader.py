import os
import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import hanlp
from tqdm import tqdm
from conf import *

cn_tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.data = [(src, tgt) for src, tgt in data]
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.rand(1) < 0.1:
            src, tgt = self.data[idx]
        else:
            src, tgt = self.data[idx]
            chinese_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
            table = str.maketrans('', '', chinese_punctuation)
            src = src.translate(table)
            tgt = tgt.translate(table)

        src_tensor = torch.tensor([self.src_vocab[token] for token in self.src_tokenizer(src)], dtype=torch.long)
        tgt_tensor = torch.tensor([self.tgt_vocab[token] for token in self.tgt_tokenizer(tgt)], dtype=torch.long)
        return src_tensor, tgt_tensor

class DataLoaderWrapper:
    def __init__(self, ext, tokenize_src, tokenize_tgt):
        self.ext = ext
        self.tokenize_src = tokenize_src
        self.tokenize_tgt = tokenize_tgt
        self.vocab_path = 'vocab.pt'

    def build_vocab(self, data, ancient_cn_tokenizer, modern_cn_tokenizer):
        # return build_vocab_from_iterator((tokenizer(sentence) for sentence, _ in data), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
        # Define paths for source and target vocab
        src_vocab_path = 'src_' + self.vocab_path
        tgt_vocab_path = 'tgt_' + self.vocab_path

        # Check if source vocab already exists
        if os.path.exists(src_vocab_path):
            # Load existing source vocab
            src_vocab = torch.load(src_vocab_path)
        else:
            # Build source vocab and save it
            src_vocab = build_vocab_from_iterator((ancient_cn_tokenizer(sentence) for sentence, _ in tqdm(data, desc="Building source vocab")), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
            torch.save(src_vocab, src_vocab_path)

        # Check if target vocab already exists
        if os.path.exists(tgt_vocab_path):
            # Load existing target vocab
            tgt_vocab = torch.load(tgt_vocab_path)
        else:
            # Build target vocab and save it
            tgt_vocab = build_vocab_from_iterator((modern_cn_tokenizer(sentence) for _, sentence in tqdm(data, desc="Building target vocab")), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
            torch.save(tgt_vocab, tgt_vocab_path)

        return src_vocab, tgt_vocab
        

    def make_dataset(self):
        # make dataset from './dataset' directory
        # the source language file is train_24_histories_m_utf8.txt, target language file is train_24_histories_c_utf8.txt
        src_file_path = 'dataset/train_24-histories_c_utf8.txt'
        tgt_file_path = 'dataset/train_24-histories_m_utf8.txt'

        with open(src_file_path, 'r', encoding='utf-8') as sf, open(tgt_file_path, 'r', encoding='utf-8') as tf:
            src_data = sf.readlines()[:100000]
            tgt_data = tf.readlines()[:100000]

            # split data into train, valid, test in 8:1:1 ratio
            train_data = list(zip(src_data[:int(len(src_data)*0.8)], tgt_data[:int(len(tgt_data)*0.8)]))
            valid_data = list(zip(src_data[int(len(src_data)*0.8):int(len(src_data)*0.9)], tgt_data[int(len(tgt_data)*0.8):int(len(tgt_data)*0.9)]))
            test_data = list(zip(src_data[int(len(src_data)*0.9):], tgt_data[int(len(tgt_data)*0.9):]))

        return train_data, valid_data, test_data

    def make_iter(self, train_data, valid_data, test_data, batch_size, src_vocab, tgt_vocab, device):
        def collate_fn(batch):
            # delete all the punctuations in the sentence
            filtered_batch = [(src, tgt) for src, tgt in batch if len(src) <= max_len and len(tgt) <= max_len]
            src_batch, tgt_batch = zip(*filtered_batch)
            src_batch = pad_sequence(src_batch, padding_value=src_vocab.get_stoi()['<pad>'], batch_first=True)
            tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab.get_stoi()['<pad>'], batch_first=True)
            return src_batch, tgt_batch

        train_loader = DataLoader(TranslationDataset(train_data, src_vocab, tgt_vocab, self.tokenize_src, self.tokenize_tgt), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(TranslationDataset(valid_data, src_vocab, tgt_vocab, self.tokenize_src, self.tokenize_tgt), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(TranslationDataset(test_data, src_vocab, tgt_vocab, self.tokenize_src, self.tokenize_tgt), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        return train_loader, valid_loader, test_loader


# Example usage:
# Define tokenizers (spacy is recommended)
def tokenize_ancient_cn(text):
    # return text.lower().split()
    return cn_tokenizer(text)

def tokenize_modern_cn(text):
    # return text.lower().split()
    return cn_tokenizer(text)


# data_loader = DataLoaderWrapper(ext=('ancient_cn', 'modern_cn'), tokenize_src=tokenize_ancient_cn, tokenize_tgt=tokenize_modern_cn)

# train_data, valid_data, test_data = data_loader.make_dataset()

# # Build vocabularies
# src_vocab = data_loader.build_vocab(train_data, tokenize_ancient_cn)
# tgt_vocab = data_loader.build_vocab(train_data, tokenize_modern_cn)

# train_loader, valid_loader, test_loader = data_loader.make_iter(train_data, valid_data, test_data, batch_size=32)













# from torchtext.legacy.data import Field, BucketIterator
# from torchtext.legacy.datasets.translation import Multi30k


# class DataLoader:
#     source: Field = None
#     target: Field = None

#     def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
#         self.ext = ext
#         self.tokenize_en = tokenize_en
#         self.tokenize_de = tokenize_de
#         self.init_token = init_token
#         self.eos_token = eos_token
#         print('dataset initializing start')

#     def make_dataset(self):
#         if self.ext == ('.de', '.en'):
#             self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)
#             self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)

#         elif self.ext == ('.en', '.de'):
#             self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)
#             self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)

#         train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
#         return train_data, valid_data, test_data

#     def build_vocab(self, train_data, min_freq):
#         self.source.build_vocab(train_data, min_freq=min_freq)
#         self.target.build_vocab(train_data, min_freq=min_freq)

#     def make_iter(self, train, validate, test, batch_size, device):
#         train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
#                                                                               batch_size=batch_size,
#                                                                               device=device)
#         print('dataset initializing done')
#         return train_iterator, valid_iterator, test_iterator