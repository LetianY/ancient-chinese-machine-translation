from hyperparameters import Hyperparams as hp
import random
import codecs
import os
import regex
from collections import Counter


def split_data(fpath, language, test_ratio=0.2):
    """
    Split the dataset into train set and test set.

    Args:
    fpath: A string. The path to the input text file.
    test_ratio: A float. The proportion of the dataset to include in the test split.
    language: "sorce" or "target"

    Returns:
    train: List of training data examples.
    test: List of test data examples.
    """
    assert language in ["source", "target"]

    with open(fpath, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - test_ratio))
    train_data = lines[:split_idx]
    test_data = lines[split_idx:]

    output_dir = "split_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_file = f"{language}_train"
    train_path = os.path.join(output_dir, train_file)
    with open(train_path, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')

    test_file = f"{language}_test"
    test_path = os.path.join(output_dir, test_file)
    with open(test_path, 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + '\n')

    return train_data, test_data


def make_vocab(dataset, fname):
    """Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed_data/fname`
    """
    text = " ".join(dataset)
    text = regex.sub("[^\s\p{L}']", "", text)
    words = text.split()
    word2cnt = Counter(words)

    if not os.path.exists("preprocessed_data"):
        os.mkdir("preprocessed_data")
    with codecs.open("preprocessed_data/{}".format(fname), "w", "utf-8") as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
                "<PAD>", "<UNK>", "<S>", "</S>"
            )
        )
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == "__main__":
    soure_train, soure_test = split_data(hp.source_tokenized_data, "source", test_ratio=0.2)
    target_train, target_test = split_data(hp.target_tokenized_data, "target", test_ratio=0.2)
    make_vocab(soure_train, "chinese.txt.vocab.tsv")
    make_vocab(target_train, "english.txt.vocab.tsv")
    print("Done")