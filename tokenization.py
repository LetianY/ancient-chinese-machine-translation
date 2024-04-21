from hyperparameters import Hyperparams as hp
import os
import hanlp
from tqdm import tqdm
import codecs

def tokenize(file_path, lang):
    """Creates source and target data."""
    if os.path.exists("tokenized_data") is False:
        os.mkdir("tokenized_data")
    with codecs.open("tokenized_data/{}.txt".format(lang), "w", "utf-8") as fout:
        if lang == "cn":
            cn_tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            with open(file_path, "r") as f:
                for line in tqdm(f.readlines(), desc="Tokenizing_cn"):
                    fout.write(" ".join(cn_tokenizer(line.strip())) + "\n")
        else:
            with open(file_path, "r") as f:
                for line in tqdm(f.readlines(), desc="Tokenizing_en"):
                    fout.write(" ".join(line.strip().split()) + "\n")


if __name__ == "__main__":
    # Tokenize
    tokenize(hp.source_data, "cn")
    tokenize(hp.target_data, "en")
    print("Done")