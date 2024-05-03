import torch
import codecs
from tqdm import tqdm
from hyperparameters import Hyperparams as hp
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def gpt2_transformer(file_path):
    """
    Extract features from the input text using GPT-2.

    Args:
    text (str): The input text.

    Returns:
    tuple: Last hidden state (features for each token).
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    custom_tokens = {'additional_special_tokens': ['[Begin]', '[Stop]']}
    tokenizer.add_special_tokens(custom_tokens)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    model.eval()  # Set the model to evaluation mode

    # If CUDA is available, use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    hidden = []
    cell = []
    with codecs.open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="Processing lines"):
            tokens = line.strip().split()
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([token_ids]).to(device)
            
            with torch.no_grad():  # No gradient needed for inference
                outputs = model(input_ids, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                last_cell_states = outputs.hidden_states[-1]
                hidden.append(last_hidden_states)
                cell.append(last_cell_states)

    return hidden, cell


encoder_last_hidden_states = gpt2_transformer(file_path=hp.source_data_c_m)
print(encoder_last_hidden_states)


###############################################################################################################


import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Extract the word token embeddings (wte)
embeddings = model.transformer.wte
embeddings.requires_grad_(False)  # Freeze the embeddings

# Example input text
input_text = "This is a test sentence."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get the embeddings for the input text
with torch.no_grad():  # Prevent gradients from being computed
    input_embeddings = embeddings(input_ids)

# Example RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out)

# Initialize RNN model
rnn_model = SimpleRNN(input_dim=input_embeddings.size(-1), hidden_dim=128, output_dim=2)

# Example usage
output = rnn_model(input_embeddings)
print(output)