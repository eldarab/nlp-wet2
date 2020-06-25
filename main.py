import torch
import numpy as np
from model import KiperwasserDependencyParser
from torch import optim
from auxiliary import convert_tree_to_list
from data import ParserDataset
from data import init_vocab_freq, init_train_freq
from torch.utils.data.dataloader import DataLoader
from train import train, draw_graphs
from eval import UAS, evaluate
import matplotlib.pyplot as plt


# data paths
data_dir = './data/'
train_path = data_dir + 'train.labeled'
test_path = data_dir + 'test.labeled'
comp_path = data_dir + 'comp.unlabeled'
train_300_path = data_dir + 'train_300.labeled'
test_300_path = data_dir + 'test_300.labeled'

# converting raw data to dedicated data objects
paths_list = [train_300_path, test_300_path]
word_dict, pos_dict = init_vocab_freq(paths_list)  # TODO https://moodle.technion.ac.il/mod/forum/discuss.php?d=522050
# train_word_dict = init_train_freq([train_path])

train_dataset = ParserDataset(word_dict, pos_dict, data_dir, 'train_300', padding=False)
train_dataloader = DataLoader(train_dataset, shuffle=True)  # batch size is 1 by default
test_dataset = ParserDataset(word_dict, pos_dict, data_dir, 'test_300', padding=False)
test_dataloader = DataLoader(test_dataset, shuffle=False)

word_vocab_size = len(train_dataset.word_idx_mappings)  # includes words from test
pos_vocab_size = len(train_dataset.pos_idx_mappings)  # includes POSs from test

# setting model hyper-parameters, in the order of appearance in the paper
word_embedding_size = 100  # Word embedding dimension
pos_embedding_size = 25    # POS tag embedding dimension
mlp_hidden_dim = 100       # Hidden units in MLP
lstm_hidden_layers = 2     # BI-LSTM Layers
encoder_hidden_size = 125  # BI-LSTM Dimensions (hidden/output)
alpha = 0.25               # alpha (for word dropout)

# setting learning hyper-parameters
epochs = 10
lr = 0.01
batch_size = 2
word_embeddings = train_dataset.word_vectors

# creating model
model = KiperwasserDependencyParser(lstm_hidden_layers=lstm_hidden_layers,
                                    word_vocab_size=word_vocab_size,
                                    word_embedding_size=word_embedding_size,
                                    pos_vocab_size=pos_vocab_size,
                                    pos_embedding_size=pos_embedding_size,
                                    encoder_hidden_size=encoder_hidden_size,
                                    mlp_hidden_dim=mlp_hidden_dim)

# moving to CUDA
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
if cuda_available:
    model.cuda()

# creating optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_list, train_acc_list, test_acc_list = train(epochs, batch_size, optimizer, train_dataset, train_dataloader,
                                                 test_dataset, model, print_epochs=True)

draw_graphs(loss_list, train_acc_list, test_acc_list)
