import os
import codecs
from tqdm import tqdm
from transformers import BertTokenizer
from hyperparameters import Hyperparams as hp

def tokenize(file_path, dataset, language):
    """
    Tokenize the input text file, truncating lines that exceed the maximum model sequence length.

    Args:
    file_path (String): The path to the input text file.
    dataset (String): Processed dataset name. Example: "pre_qin" and "24_history".
    language (String): Language of the source data. Example: "c" for Chinese, "e" for English.
    """
    
    # Load tokenizer with specific pre-trained model
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-ancient")
    # Add custom special tokens
    custom_tokens = {'additional_special_tokens': ['[Begin]', '[Stop]']}
    tokenizer.add_special_tokens(custom_tokens)

    # Maximum token length for the model
    max_length = 512  # Model's maximum input size

    # Create a directory to save the tokenized data
    if not os.path.exists("data_tokenized_gpt2"):
        os.mkdir("data_tokenized_gpt2")

    output_file_path = f"data_tokenized_gpt2/{dataset}_{language}.txt"
    
    # Tokenize the input text file
    with codecs.open(output_file_path, "w", "utf-8") as fout:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc=f"Tokenizing {language}"):
                # Wrap each line with custom special tokens
                line_with_tokens = f"[Begin] {line.strip()} [Stop]"
                # Tokenize and convert to IDs without default special tokens handling
                token_ids = tokenizer.encode(line_with_tokens, add_special_tokens=False)
                # Convert token_ids back to tokens for readability
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                fout.write(" ".join(tokens) + "\n")

print("Processing c-m dataset...")
# Example usage (make sure to define these or adapt to your actual use)
source_data_name = hp.source_data_c_m
target_data_name = hp.target_data_c_m
dataset_name = '24_history'
source_lang = 'c'
target_lang = 'm'

tokenize(file_path=source_data_name, dataset=dataset_name, language=source_lang)
tokenize(file_path=target_data_name, dataset=dataset_name, language=target_lang)
