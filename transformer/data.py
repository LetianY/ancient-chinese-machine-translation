from conf import *
# from util.data_loader import DataLoader
from util.data_loader import *
# from util.tokenizer import Tokenizer

# tokenizer = Tokenizer()
loader = DataLoaderWrapper(ext=('ancient_cn', 'modern_cn'), 
                           tokenize_src=tokenize_ancient_cn, 
                           tokenize_tgt=tokenize_modern_cn
                           )

train, valid, test = loader.make_dataset()

src_vocab, tgt_vocab = loader.build_vocab(train+valid+test, tokenize_ancient_cn, tokenize_modern_cn)
# tgt_vocab = loader.build_vocab(train+valid+test, )

src_vocab.set_default_index(src_vocab.get_stoi()['<unk>'])
tgt_vocab.set_default_index(tgt_vocab.get_stoi()['<unk>'])

train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                        batch_size=batch_size, src_vocab=src_vocab, tgt_vocab=tgt_vocab, device=device)

src_pad_idx = src_vocab.get_stoi()['<pad>']
trg_pad_idx = tgt_vocab.get_stoi()['<pad>']
trg_sos_idx = tgt_vocab.get_stoi()['<sos>']

enc_voc_size = len(src_vocab)
dec_voc_size = len(tgt_vocab)
