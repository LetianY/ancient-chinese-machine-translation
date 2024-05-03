import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from hyperparameters import Hyperparams as hp
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# def gpt2_transformer(file_path):
#     """
#     Extract features from the input text using GPT-2.

#     Args:
#     text (str): The input text.

#     Returns:
#     tuple: Last hidden state (features for each token).
#     """
#     # Load tokenizer and model
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     custom_tokens = {'additional_special_tokens': ['[Begin]', '[Stop]']}
#     tokenizer.add_special_tokens(custom_tokens)

#     model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-ancient")
#     model.resize_token_embeddings(len(tokenizer))

#     model.eval()  # Set the model to evaluation mode

#     # If CUDA is available, use GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     hidden = []
#     cell = []
#     with codecs.open(file_path, 'r', encoding='utf-8') as file:
#         discount = 0
#         for line in tqdm(file.readlines(), desc="Processing lines"):
#             if discount <= 100:            
#                 tokens = line.strip().split()
#                 token_ids = tokenizer.convert_tokens_to_ids(tokens)
#                 input_ids = torch.tensor([token_ids]).to(device)
                
#                 with torch.no_grad():  # No gradient needed for inference
#                     outputs = model(input_ids, output_hidden_states=True)
#                     last_hidden_states = outputs.hidden_states[-1]
#                     discount += 1
#             else:
#                 break

#     last_hidden_states = last_hidden_states.transpose(0, 1)  # Switch batch_size and seq_length
#     return last_hidden_states, last_hidden_states.clone()  # Clone to mimic cell state


class EncoderRNN(nn.Module):
    """
    EncoderRNN is a part of the sequence-to-sequence architecture that encodes the input sequence
    into a context vector. This context vector (final hidden state of the LSTM) captures the essence
    of the input data and is used by the decoder as the initial hidden state.
    """
    def __init__(self, embed_size, hidden_size):
        """        
        Parameters:
            input_size (int): The size of the input vocabulary.
            embed_size (int): The dimensionality of the embedding space.
            hidden_size (int): The number of features in the hidden state of the LSTM.
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # self.embed_size = embed_size
        # self.input_size = input_size
        # self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embed_size)
        # self.embed_size = embed_size # Modif it to the length of the embedding
        self.lstm = nn.LSTM(embed_size, self.hidden_size, batch_first=True)

    def forward(self, input_embeddings):
        # embedded = self.embedding(input)
        _, (hidden, cell) = self.lstm(input_embeddings)
        return hidden, cell


class DecoderRNN(nn.Module):
    """
    DecoderRNN is used in the sequence-to-sequence architecture to decode the encoded information.
    Starting from the context vector provided by the EncoderRNN, it generates the output sequence
    one token at a time.
    
    """
    def __init__(self, embed_size, hidden_size, output_size):
        """        
        Parameters:
            embed_size (int): The dimensionality of the embedding space.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            output_size (int): The size of the output vocabulary.
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        output = self.embedding(input)
        output = torch.relu(output)
        output, hidden = self.lstm(output, (hidden, cell))
        output = self.out(output)
        return output, hidden


# Define the combined model including both EncoderRNN and DecoderRNN
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        # Implement the forward pass calling the encoder and decoder
        hidden, cell = self.encoder.initHidden(hp.batch_size)
        hidden, cell = self.encoder(source)
        # hidden, cell = gpt2_transformer(file_path=hp.source_data_c_m)
        output, _ = self.decoder(target, (hidden, cell))
        return output
    

def metric(output, y):
    # device
    dec_voc = output.shape[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(output.shape, y.shape)
    probs = F.softmax(output, dim=-1).view(-1, dec_voc)
    _, preds = torch.max(output, -1)
    # print(output.shape)
    # probs = output.view(-1, output.size()[1])
    istarget = (1.0 - y.eq(0.0).float()).view(-1)
    acc = torch.sum(
        preds.eq(y).float().view(-1) * istarget
    ) / torch.sum(istarget)

    # Loss
    y_onehot = torch.zeros(
        output.size()[0] * output.size()[1], dec_voc
    ).to(device)
    y_onehot = y_onehot.scatter_(1, y.view(-1, 1).data, 1)

    loss = -torch.sum(y_onehot * torch.log(probs), dim=-1)
    mean_loss = torch.sum(loss * istarget) / torch.sum(istarget)

    return mean_loss, preds, acc