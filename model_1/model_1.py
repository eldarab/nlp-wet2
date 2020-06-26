from chu_liu_edmonds import decode_mst, NLLLoss
from torch import nn
import torch
from warnings import warn


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, lstm_hidden_layers, pos_vocab_size, pos_embedding_size, encoder_hidden_size, mlp_hidden_dim,
                 word_vocab_size=None, word_embedding_size=None, word_embeddings=None):
        super(KiperwasserDependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            warn('Using CPU!')
        if word_embeddings:
            self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        else:
            self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_size)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_size)
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = nn.LSTM(input_size=self.hidden_dim, hidden_size=encoder_hidden_size,
                               num_layers=lstm_hidden_layers, bidirectional=True)
        self.edge_scorer = MLPScorer(4 * encoder_hidden_size, mlp_hidden_dim)
        self.decoder = decode_mst
        self.loss_function = NLLLoss

    def forward(self, sentence, calculate_loss=True):
        word_idx_tensor, pos_idx_tensor, _, true_tree_heads = sentence

        word_idx_tensor = torch.squeeze(word_idx_tensor.to(self.device))
        pos_idx_tensor = torch.squeeze(pos_idx_tensor.to(self.device))

        words_embedded = self.word_embedding(word_idx_tensor)                                       # [seq_length, word_embedding_size]
        poss_embedded = self.pos_embedding(pos_idx_tensor)                                          # [seq_length, pos_embedding_size]
        embeds = torch.cat((words_embedded, poss_embedded), dim=1).view(-1, 1, self.hidden_dim)     # [seq_length, batch_size, hidden_dim]
        lstm_out, _ = self.encoder(embeds)                                                          # [seq_length, batch_size, 2*hidden_dim]
        score_matrix = self.edge_scorer(lstm_out)                                                   # [seq_length, seq_length]
        predicted_tree, _ = self.decoder(score_matrix.detach().numpy(), score_matrix.shape[0], has_labels=False)
        if calculate_loss:
            loss = self.loss_function(score_matrix, true_tree_heads)
        else:
            loss = None

        return loss, predicted_tree


class MLPScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPScorer, self).__init__()
        self.W1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.activation = nn.Tanh()  # if we use torch.nn.functional.tanh we get warned :(
        self.W2 = nn.Linear(in_features=hidden_dim, out_features=1)  # output is MLP score

    def forward(self, input):
        input = input.view(input.shape[0], input.shape[2])
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