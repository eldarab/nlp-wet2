

# ▓█████  ██▓    ▓█████▄  ▄▄▄       ██▀███             ▄▄▄       ▄▄▄▄    ██▀███   ▄▄▄       ██░ ██  ▄▄▄       ███▄ ▄███▓
# ▓█   ▀ ▓██▒    ▒██▀ ██▌▒████▄    ▓██ ▒ ██▒          ▒████▄    ▓█████▄ ▓██ ▒ ██▒▒████▄    ▓██░ ██▒▒████▄    ▓██▒▀█▀ ██▒
# ▒███   ▒██░    ░██   █▌▒██  ▀█▄  ▓██ ░▄█ ▒          ▒██  ▀█▄  ▒██▒ ▄██▓██ ░▄█ ▒▒██  ▀█▄  ▒██▀▀██░▒██  ▀█▄  ▓██    ▓██░
# ▒▓█  ▄ ▒██░    ░▓█▄   ▌░██▄▄▄▄██ ▒██▀▀█▄            ░██▄▄▄▄██ ▒██░█▀  ▒██▀▀█▄  ░██▄▄▄▄██ ░▓█ ░██ ░██▄▄▄▄██ ▒██    ▒██
# ░▒████▒░██████▒░▒████▓  ▓█   ▓██▒░██▓ ▒██▒           ▓█   ▓██▒░▓█  ▀█▓░██▓ ▒██▒ ▓█   ▓██▒░▓█▒░██▓ ▓█   ▓██▒▒██▒   ░██▒
# ░░ ▒░ ░░ ▒░▓  ░ ▒▒▓  ▒  ▒▒   ▓▒█░░ ▒▓ ░▒▓░           ▒▒   ▓▒█░░▒▓███▀▒░ ▒▓ ░▒▓░ ▒▒   ▓▒█░ ▒ ░░▒░▒ ▒▒   ▓▒█░░ ▒░   ░  ░
#  ░ ░  ░░ ░ ▒  ░ ░ ▒  ▒   ▒   ▒▒ ░  ░▒ ░ ▒░            ▒   ▒▒ ░▒░▒   ░   ░▒ ░ ▒░  ▒   ▒▒ ░ ▒ ░▒░ ░  ▒   ▒▒ ░░  ░      ░
#    ░     ░ ░    ░ ░  ░   ░   ▒     ░░   ░             ░   ▒    ░    ░   ░░   ░   ░   ▒    ░  ░░ ░  ░   ▒   ░      ░
#    ░  ░    ░  ░   ░          ░  ░   ░                     ░  ░ ░         ░           ░  ░ ░  ░  ░      ░  ░       ░
#                 ░                                                   ░


#  ██░ ██  ▄▄▄      ▓█████▄  ▄▄▄       ██▀███              ██████  ██▓ ███▄    █  ▄▄▄       ██▓
# ▓██░ ██▒▒████▄    ▒██▀ ██▌▒████▄    ▓██ ▒ ██▒          ▒██    ▒ ▓██▒ ██ ▀█   █ ▒████▄    ▓██▒
# ▒██▀▀██░▒██  ▀█▄  ░██   █▌▒██  ▀█▄  ▓██ ░▄█ ▒          ░ ▓██▄   ▒██▒▓██  ▀█ ██▒▒██  ▀█▄  ▒██▒
# ░▓█ ░██ ░██▄▄▄▄██ ░▓█▄   ▌░██▄▄▄▄██ ▒██▀▀█▄              ▒   ██▒░██░▓██▒  ▐▌██▒░██▄▄▄▄██ ░██░
# ░▓█▒░██▓ ▓█   ▓██▒░▒████▓  ▓█   ▓██▒░██▓ ▒██▒          ▒██████▒▒░██░▒██░   ▓██░ ▓█   ▓██▒░██░
#  ▒ ░░▒░▒ ▒▒   ▓▒█░ ▒▒▓  ▒  ▒▒   ▓▒█░░ ▒▓ ░▒▓░          ▒ ▒▓▒ ▒ ░░▓  ░ ▒░   ▒ ▒  ▒▒   ▓▒█░░▓
#  ▒ ░▒░ ░  ▒   ▒▒ ░ ░ ▒  ▒   ▒   ▒▒ ░  ░▒ ░ ▒░          ░ ░▒  ░ ░ ▒ ░░ ░░   ░ ▒░  ▒   ▒▒ ░ ▒ ░
#  ░  ░░ ░  ░   ▒    ░ ░  ░   ░   ▒     ░░   ░           ░  ░  ░   ▒ ░   ░   ░ ░   ░   ▒    ▒ ░
#  ░  ░  ░      ░  ░   ░          ░  ░   ░                     ░   ░           ░       ░  ░ ░
#                    ░


import pickle
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from eval import evaluate, evaluate_old
from data import init_vocab_freq, ParserDataset
from train import train_model, draw_graphs
from matplotlib import pyplot as plt

data_dir = 'data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
combined_filename = 'combined.labeled'
train_200_filename = 'train_200.labeled'
test_200_filename = 'test_200.labeled'
comp_filename = 'comp.unlabeled'


train_acc_list = [0.8496, 0.8733, 0.8711, 0.8725]
test_acc_list = [0.6743, 0.8956, 0.936, 0.9551]
loss_list = [0.597, 0.339, 0.2061, 0.1414]

plt.plot(train_acc_list, c="red", label="Train accuracy")
plt.plot(test_acc_list, c="green", label="Test accuracy")
plt.title("Train and test accuracies (UAS) vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("accuracy %")
plt.legend()
plt.show()

plt.plot(loss_list, c="blue", label="Loss")
plt.title("Loss vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()


# paths_list = [data_dir + train_filename, data_dir + test_filename]
# word_dict, pos_dict = init_vocab_freq(paths_list, lowercase=True)
# test_dataset = ParserDataset(word_dict, pos_dict, data_dir, test_filename, lowercase=True)
# test_dataloader = DataLoader(test_dataset, shuffle=False)
# with open('dumps/model-glove12/epoch6.eh', 'rb') as f:
#     loaded_model = torch.load(f)
# loaded_model.to('cuda')
#
# print('accuracy old: ', evaluate_old(loaded_model, test_dataloader))
# print('accuracy new: ', evaluate(loaded_model, test_dataloader))

# model_name = 'base-model'
# _, loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
#                                                           data_dir=data_dir,
#                                                           filenames=[train_filename, test_filename],
#                                                           word_embedding_size=100,
#                                                           pos_embedding_size=25,
#                                                           mlp_hidden_dim=100,
#                                                           lstm_hidden_layers=2,
#                                                           encoder_hidden_size=125,
#                                                           alpha=0.15,
#                                                           word_embeddings=None,
#                                                           lowercase=False,
#                                                           epochs=10,
#                                                           lr=0.01,
#                                                           activation=nn.Tanh(),
#                                                           batch_size=50,
#                                                           CUDA=True,
#                                                           print_epochs=True,
#                                                           save_dir='dumps/')
#
with open('dumps/model-glove12_graphs.pkl', 'rb') as f:
    loss_list, train_acc_list, test_acc_list = pickle.load(f)
draw_graphs(loss_list, train_acc_list, test_acc_list)
# print('finished training ', model_name)

