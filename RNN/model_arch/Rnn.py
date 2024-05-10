import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from hyperparameters import Hyperparams as hp
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
        self.lstm = nn.LSTM(embed_size, self.hidden_size, batch_first=True)
        # self.lstm = nn.LSTM(embed_size, self.hidden_size)

    def forward(self, input_embeddings):
        output, (hidden, cell) = self.lstm(input_embeddings)
        # hidden = hidden.transpose(1, 0, 2)
        # hidden = torch.transpose(hidden, 0, 1)
        # cell = cell.transpose(1, 0, 2)
        # cell = torch.transpose(cell, 0, 1)
        if len(hidden.shape) == 2:
            # if hidden or cell only have two dimensions, then add one more dimension representing batch
            hidden = hidden.unsqueeze(1)
            cell = cell.unsqueeze(1)
        # print("________BEGIN_ENCODER_________")
        # print("Hidden:", hidden.shape)
        # print("Cell:", cell.shape)
        # print("______________________________")
        return hidden, cell
    
    # def initHidden(self, batch_size):
    #     # Initialize hidden and cell states with the correct batch size
    #     return (torch.zeros(batch_size, 1, self.hidden_size),
    #             torch.zeros(batch_size, 1, self.hidden_size))


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
        """        
        Parameters:
            input: the target language
            hidden: the last hidden state from Encoder
            cell: the last cell state from Encoder
        """
        # print('input: ', input.shape)
        # if len(input.shape) > 2:
        #     input = input.squeeze()
        target_embedding = self.embedding(input)
        target_embedding = torch.relu(target_embedding)
        # print("target_embedding", target_embedding.shape)
        # print("hidden", hidden.shape)
        # print("cell", cell.shape)
        output, hidden = self.lstm(target_embedding, (hidden, cell))
        output = self.out(output)
        # print("output", output.shape)
        # print("________END_DECODER________")
        return output, hidden


# Define the combined model including both EncoderRNN and DecoderRNN
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        # Implement the forward pass calling the encoder and decoder
        # hidden, cell = self.encoder.initHidden(hp.batch_size)
        hidden, cell = self.encoder(source)
        output, _ = self.decoder(target, hidden, cell)
        return output
    

def sequence_loss(output, target, pad_index):
    """
    Compute the Cross-Entropy Loss for a sequence of logits.

    Args:
    output (torch.Tensor): Logits from the model (batch_size, seq_length, vocab_size)
    target (torch.Tensor): Ground-truth indices (batch_size, seq_length)
    
    Returns:
    torch.Tensor: The loss value.
    """
    # Reshape output to (batch_size * seq_length, vocab_size)
    # Reshape target to (batch_size * seq_length)
    output = output.view(-1, output.shape[-1])
    target = target[:, 1:].reshape(-1)
    
    # Compute cross-entropy loss, ignore index=-100 to mask out padding tokens if any
    loss = F.cross_entropy(output, target, ignore_index=pad_index)
    
    return loss


# def metric(output, y):
#     # device
#     dec_voc = output.shape[-1]
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # print(output.shape, y.shape)
#     probs = F.softmax(output, dim=-1).view(-1, dec_voc)
#     _, preds = torch.max(output, -1)
#     # print(output.shape)
#     # probs = output.view(-1, output.size()[1])
#     istarget = (1.0 - y.eq(0.0).float()).view(-1)
#     acc = torch.sum(
#         preds.eq(y).float().view(-1) * istarget
#     ) / torch.sum(istarget)

#     # Loss
#     y_onehot = torch.zeros(
#         output.size()[0] * output.size()[1], dec_voc
#     ).to(device)
#     y_onehot = y_onehot.scatter_(1, y.view(-1, 1).data, 1)

#     loss = -torch.sum(y_onehot * torch.log(probs), dim=-1)
#     mean_loss = torch.sum(loss * istarget) / torch.sum(istarget)

#     return mean_loss, preds, acc