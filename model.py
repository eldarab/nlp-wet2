from torch import nn


class DnnParser(nn.Module):
    def __init__(self, word_vec_dim, lstm_hidden_size, mlp_hidden_size, mlp_output_size, lstm_hidden_layers=2):
        super(DnnParser, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=word_vec_dim, hidden_size=lstm_hidden_size,
                               num_layers=lstm_hidden_layers, bidirectional=True)
        self.score_mlp = nn.Linear(in_features=2 * lstm_hidden_size, out_features=mlp_hidden_size), \
                         nn.Linear(in_features=mlp_hidden_size, out_features=mlp_output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.bi_lstm(x)             # x.size() -> [batch_size, lstm_hidden_size]
        x = self.score_mlp[0](x)        # x.size() -> [batch_size, mlp_hidden_size]
        x = self.activation(x)          # x.size() -> [batch_size, mlp_hidden_size]
        x = self.score_mlp[1](x)        # x.size() ->
        return x


def NLLLoss(D):
    for x, y in D:
        pass