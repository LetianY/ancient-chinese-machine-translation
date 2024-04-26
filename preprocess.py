from hyperparameters import Hyperparams as hp
import random
import codecs
import os
import regex
from collections import Counter
import hanlp
from tqdm import tqdm

def tokenize(file_path, dataset, language):
    """
    Tokenize the input text file.

    Args:
    file_path (String): The path to the input text file.
    token_name (String): corresponding tokenized output file name
    dataset (String): processed dataset name. "pre_qin" and "24_history"
    source_lang (String): Language of the source data. "c" or "m" for Chinese, "e" for English.
    """
    # Create a directory to save the tokenized data
    if os.path.exists("data_tokenized") is False:
        os.mkdir("data_tokenized")
    
    # Tokenize the input text file
    with codecs.open(f"data_tokenized/{dataset}_{language}.txt", "w", "utf-8") as fout:
        if language == "e":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f.readlines(), desc="Tokenizing_en"):
                    fout.write(" ".join(line.strip().split()) + "\n")
        else:
            # Alternative: COARSE_ELECTRA_SMALL_ZH
            cn_tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f.readlines(), desc="Tokenizing_cn"):
                    fout.write(" ".join(cn_tokenizer(line.strip())) + "\n")

def split_data(fpath, dataset, source_lang, target_lang, test_ratio=0.2):
    """
    Split the dataset into train set and test set.

    Args:
    fpath (String): The path to the tokenized data.
    token_name (String): corresponding tokenized output file name
    dataset (String): processed dataset name. "pre_qin" and "24_history"
    source_lang (String): Language of the source data. "c" or "m" for Chinese, "e" for English.
    target_lang (String): Language of the target data. "c" or "m" for Chinese, "e" for English.
    test_ratio (float): The ratio of the test set.

    Returns:
    source_train_data (List): List of training sentences in the source dataset.
    source_test_data (List): List of testing sentences in the source dataset.
    target_train_data (List): List of training sentences in the target dataset.
    target_test_data (List): List of testing sentences in the target dataset.
    """
    # Read the tokenized source and target data
    with codecs.open(fpath + f"/{dataset}_{source_lang}.txt", "r", "utf-8") as f:
        source_data = f.readlines()
    with codecs.open(fpath + f"/{dataset}_{target_lang}.txt", "r", "utf-8") as f:
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
    if source_lang=='m' and target_lang=='e':
        source_train_name = hp.source_train_m_e
        source_test_name = hp.source_test_m_e
        target_train_name = hp.target_train_m_e
        target_test_name = hp.target_test_m_e
    elif source_lang=='c' and target_lang=='m':
        source_train_name = hp.source_train_c_m
        source_test_name = hp.source_test_c_m
        target_train_name = hp.target_train_c_m
        target_test_name = hp.target_test_c_m
    else:
        raise ValueError("Incorrect source_lang or target_lang!")

    with codecs.open(source_train_name, "w", "utf-8") as f:
        f.write("".join(source_train_data))
    with codecs.open(source_test_name, "w", "utf-8") as f:
        f.write("".join(source_test_data))
    with codecs.open(target_train_name, "w", "utf-8") as f:
        f.write("".join(target_train_data))
    with codecs.open(target_test_name, "w", "utf-8") as f:
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
    task_types = ['m_e', 'c_m']
    for task_type in task_types:
        # Process Pre-Qin + ZiZhiTongJian Dataset
        if task_type == 'm_e':
            print("Processing m-e dataset...")
            source_data_name = hp.source_data_m_e
            target_data_name = hp.target_data_m_e
            dataset_name = 'pre_qin'
            source_lang = 'm'
            target_lang = 'e'
        # Process 24 History Dataset
        else:
            print("Processing c-m dataset...")
            source_data_name = hp.source_data_c_m
            target_data_name = hp.target_data_c_m
            dataset_name = '24_history'
            source_lang = 'c'
            target_lang = 'm'

        # Tokenize the source and target data
        print("\t Tokenizing the source and target data...")
        tokenize(file_path=source_data_name, dataset=dataset_name, language=source_lang)
        tokenize(file_path=target_data_name, dataset=dataset_name, language=target_lang)

        # Split the dataset into train set and test set
        print("\t Splitting the dataset into train set and test set...")
        source_train, source_test, target_train, target_test = \
            split_data(fpath='./data_tokenized', dataset=dataset_name,
                       source_lang=source_lang, target_lang=target_lang, test_ratio=0.2)

        # Construct vocabulary
        print("\t Constructing vocabulary...")
        make_vocab(source_train + source_test, fname=f"{dataset_name}_{source_lang}.txt.vocab.tsv")
        make_vocab(target_train + target_test, fname=f"{dataset_name}_{target_lang}.txt.vocab.tsv")

    print("Done")