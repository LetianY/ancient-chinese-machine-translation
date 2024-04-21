from hyperparameters import Hyperparams as hp
import random
import codecs
import os
import regex
from collections import Counter
import hanlp
from tqdm import tqdm

def tokenize(file_path, language):
    """
    Tokenize the input text file.

    Args:
    file_path (String): The path to the input text file.
    lang (String): Language of the text file. "cn" for Chinese, "en" for English.
    """
    # Create a directory to save the tokenized data
    if os.path.exists("data_tokenized") is False:
        os.mkdir("data_tokenized")
    
    # Tokenize the input text file
    with codecs.open("data_tokenized/{}.txt".format(language), "w", "utf-8") as fout:
        if language == "cn":
            cn_tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            with open(file_path, "r") as f:
                for line in tqdm(f.readlines(), desc="Tokenizing_cn"):
                    fout.write(" ".join(cn_tokenizer(line.strip())) + "\n")
        else:
            with open(file_path, "r") as f:
                for line in tqdm(f.readlines(), desc="Tokenizing_en"):
                    fout.write(" ".join(line.strip().split()) + "\n")

def split_data(fpath, source_lang='cn', target_lang='en', test_ratio=0.2):
    """
    Split the dataset into train set and test set.

    Args:
    fpath (String): The path to the tokenized data.
    source_lang (String): Language of the source data. "cn" for Chinese, "en" for English.
    target_lang (String): Language of the target data. "cn" for Chinese, "en" for English.
    test_ratio (float): The ratio of the test set.

    Returns:
    source_train_data (List): List of training sentences in the source dataset.
    source_test_data (List): List of testing sentences in the source dataset.
    target_train_data (List): List of training sentences in the target dataset.
    target_test_data (List): List of testing sentences in the target dataset.
    """
    # Read the tokenized source and target data
    with codecs.open(fpath + "/{}.txt".format(source_lang), "r", "utf-8") as f:
        source_data = f.readlines()
    with codecs.open(fpath + "/{}.txt".format(target_lang), "r", "utf-8") as f:
        target_data = f.readlines()
    
    # Generate random indices for splitting the dataset
    indices = list(range(len(source_data)))
    random.shuffle(indices)
    test_size = int(len(source_data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split the dataset into train set and test set
    source_train_data = [source_data[i] for i in train_indices]
    source_test_data = [source_data[i] for i in test_indices]
    target_train_data = [target_data[i] for i in train_indices]
    target_test_data = [target_data[i] for i in test_indices]

    # Save the train set and test set
    with codecs.open(hp.source_train, "w", "utf-8") as f:
        f.write("".join(source_train_data))
    with codecs.open(hp.source_test, "w", "utf-8") as f:
        f.write("".join(source_test_data))
    with codecs.open(hp.target_train, "w", "utf-8") as f:
        f.write("".join(target_train_data))
    with codecs.open(hp.target_test, "w", "utf-8") as f:
        f.write("".join(target_test_data))

    return source_train_data, source_test_data, target_train_data, target_test_data

def make_vocab(dataset, fname):
    """Construct vocabulary.
    
    Args:
        dataset (List): List of sentences.
        fname (String): Name of the file to save the vocabulary.
    
    Writes vocabulary line by line to `data_vocab/fname`
    """
    text = " ".join(dataset)
    text = regex.sub("[^\s\p{L}']", "", text)
    words = text.split()
    word2cnt = Counter(words)

    if not os.path.exists("data_vocab"):
        os.mkdir("data_vocab")

    with codecs.open("data_vocab/{}".format(fname), "w", "utf-8") as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
                "<PAD>", "<UNK>", "<S>", "</S>"
            )
        )
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == "__main__":
    # Tokenize the source and target data
    print("Tokenizing the source and target data...")
    tokenize(hp.source_data, language="cn")
    tokenize(hp.target_data, language="en")

    # Split the dataset into train set and test set
    print("Splitting the dataset into train set and test set...")
    source_train, source_test, target_train, target_test = split_data('./data_tokenized', source_lang='cn', target_lang='en', test_ratio=0.2)
    
    # Construct vocabulary
    print("Constructing vocabulary...")
    make_vocab(source_train + source_test, "chinese.txt.vocab.tsv")
    make_vocab(target_train + target_test, "english.txt.vocab.tsv")
    print("Done")