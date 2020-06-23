from chu_liu_edmonds import decode_mst
from torch import nn
import torch
import numpy as np


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, word_vec_dim, word_embedding_hidden_size, lstm_hidden_layers, pos_vec_dim, encoder_hidden_size,
                 pos_embedding_hidden_size):
        super(KiperwasserDependencyParser, self).__init__()

        # embedding layer for words (can be new or pre-trained - word2vec/glove)
        self.word_embedding = nn.LSTM(input_size=word_vec_dim, hidden_size=word_embedding_hidden_size,
                                      num_layers=lstm_hidden_layers, bidirectional=True)

        # embedding layer for POS tags
        self.pos_embedding = nn.LSTM(input_size=pos_vec_dim, hidden_size=pos_embedding_hidden_size,
                                     num_layers=lstm_hidden_layers, bidirectional=True)

        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim

        # BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.encoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=encoder_hidden_size,
                               num_layers=lstm_hidden_layers, bidirectional=True)

        # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.edge_scorer =
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function =  # Implement the loss function described above

    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, _, true_tree_heads = sentence  # TODO turn _ to sentence_len if turned out to be useful

        # Pass word_idx and pos_idx through their embedding layers
        sentence_word_embedded = self.word_embedding(word_idx_tensor)
        sentence_pos_embedded = self.pos_embedding(pos_idx_tensor)

        # Concat both embedding outputs
        word_pos_concat = torch.cat((sentence_word_embedded, sentence_pos_embedded), dim=0)

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        word_pos_hidden_rep = self.encoder(word_pos_concat)

        # Get score for each possible edge in the parsing graph, construct score matrix

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix

        # Calculate the negative log likelihood loss described above

        # return loss, predicted_tree

def softmax(s, x, theta):
    """
    Calculates the probability of a word to be a head word, TODO given the modifier??.
    :param s: S^i_{h,m}
    :param x: X^i
    :param theta: theta
    :return:
    """


# TODO should we use the predefined NLLLoss of pytorch?
def NLLLoss(D, theta):
    loss = nn.NLLLoss()
