from auxiliary import NLLLoss
from chu_liu_edmonds import decode_mst
from torch import nn
import torch
from warnings import warn


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, lstm_hidden_layers, pos_vocab_size, pos_embedding_size, encoder_hidden_size, mlp_hidden_dim,
                 word_vocab_size=None, word_embedding_size=None, word_embeddings=None,
                 loss_function=NLLLoss, activation_function=nn.Tanh()):
        super(KiperwasserDependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation = activation_function
        if self.device == "cpu":
            warn('Using CPU!')
        if (word_vocab_size is None or word_embedding_size is None) and word_embeddings is None:
            raise Exception('No word embeddings have been given to the model')

        if word_embeddings:
            self.pre_trained_word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        else:
            self.pre_trained_word_embedding = None
        if word_vocab_size and word_embedding_size:
            self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_size)
        else:
            self.word_embedding = None
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_size)

        self.hidden_dim = self.get_embedding_dim()
        self.encoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=encoder_hidden_size,
                               num_layers=lstm_hidden_layers, bidirectional=True)

        # self.edge_scorer = MLPScorer(4 * encoder_hidden_size, mlp_hidden_dim, activation_function)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=4 * encoder_hidden_size, out_features=mlp_hidden_dim),
            self.activation,
            nn.Linear(in_features=mlp_hidden_dim, out_features=1)
        )
        self.decoder = decode_mst
        self.loss_function = loss_function

    def forward(self, sentence, calculate_loss=True):
        word_idx_tensor, pos_idx_tensor, _, true_tree_heads = sentence  # TODO padding

        word_idx_tensor = torch.squeeze(word_idx_tensor.to(self.device))
        pos_idx_tensor = torch.squeeze(pos_idx_tensor.to(self.device))

        words_embedded = self.embed_words(word_idx_tensor)                                          # [seq_length, word_embedding_size]
        poss_embedded = self.pos_embedding(pos_idx_tensor)                                          # [seq_length, pos_embedding_size]
        embeds = torch.cat((words_embedded, poss_embedded), dim=1).view(-1, 1, self.hidden_dim)     # [seq_length, batch_size, hidden_dim]
        lstm_out, _ = self.encoder(embeds)                                                          # [seq_length, batch_size, 2*hidden_dim]
        # score_matrix = self.edge_scorer(lstm_out)                                                 # [seq_length, seq_length]
        score_matrix = self.generate_score_matrix(lstm_out)
        predicted_tree, _ = self.decoder(score_matrix.detach().numpy(), score_matrix.shape[0], has_labels=False)

        if calculate_loss:
            loss = self.loss_function(score_matrix, true_tree_heads)
        else:
            loss = None
        return loss, predicted_tree

    def embed_words(self, words_idx_tensor):
        word_embedding = None
        if self.pre_trained_word_embedding is not None and self.word_embedding is not None:
            word_embedding = torch.cat(self.pre_trained_word_embedding(words_idx_tensor),
                                       self.word_embedding(words_idx_tensor))
        elif self.pre_trained_word_embedding is not None:
            word_embedding = self.pre_trained_word_embedding(words_idx_tensor)
        elif self.word_embedding is not None:
            word_embedding = self.word_embedding(words_idx_tensor)
        return word_embedding

    def get_embedding_dim(self):
        dim = self.pos_embedding.embedding_dim
        if self.word_embedding is not None:
            dim += self.word_embedding.embedding_dim
        if self.pre_trained_word_embedding is not None:
            dim += self.pre_trained_word_embedding.embedding_dim
        return dim

    def generate_score_matrix(self, word_vectors):
        word_vectors = torch.squeeze(word_vectors)
        n = word_vectors.shape[0]
        score_matrix = torch.empty(n, n)
        for i in range(n):
            for j in range(n):
                v = torch.cat((word_vectors[i], word_vectors[j])).unsqueeze(0)
                score_matrix[i][j] = self.MLP(v)
        return score_matrix


# This nn has been replaced by a sequential in the main nn
class MLPScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_function):
        super(MLPScorer, self).__init__()
        self.W1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.activation = activation_function
        self.W2 = nn.Linear(in_features=hidden_dim, out_features=1)  # output is MLP score

    def forward(self, input):
        input = torch.squeeze(input)
        n = input.shape[0]
        score_matrix = torch.empty(n, n)
        for i in range(n):
            for j in range(n):
                v = torch.cat((input[i], input[j])).unsqueeze(0)
                x = self.W1(v)
                x = self.activation(x)
                x = self.W2(x)
                score_matrix[i][j] = x
        return score_matrix
