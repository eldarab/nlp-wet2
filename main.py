import pickle

import torch
from torch.utils.data.dataloader import DataLoader

from eval import evaluate, predict_data
from data_1 import ParserDataset, init_vocab_freq
from train import train_model, draw_graphs


def hp_tuning(word_embedding_sizes, pos_embedding_sizes, mlp_hidden_dims,
                lstm_hidden_layerss, alphas, lrs):
    model_index = 10
    best_acc = 0
    best_model = -1  # meaningless
    for lstm_hidden_layers in lstm_hidden_layerss:
        loss_list, train_acc_list, test_acc_list = train_model(model_name='model' + str(model_index),
                                                               data_dir=data_dir,
                                                               filenames=[train_filename, test_filename],
                                                               word_embedding_size=word_embedding_sizes[0],
                                                               pos_embedding_size=pos_embedding_sizes[0],
                                                               mlp_hidden_dim=mlp_hidden_dims[0],
                                                               lstm_hidden_layers=lstm_hidden_layers,
                                                               encoder_hidden_size=word_embedding_sizes[0] + pos_embedding_sizes[0],
                                                               alpha=alphas[0],
                                                               word_embeddings=None,
                                                               epochs=4,
                                                               lr=lrs[0],
                                                               batch_size=50,
                                                               CUDA=True,
                                                               print_epochs=True,
                                                               save_dir='./dumps/')
        draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/model' + str(model_index) + '_graphs.pkl')
        if test_acc_list[-1] > best_acc:
            best_acc = test_acc_list[-1]
            best_model = model_index


data_dir = './data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
comp_filename = 'comp.unlabeled'
train_300_filename = 'train_300.labeled'
test_300_filename = 'test_300.labeled'


with open('./dumps/model10/epoch5.eh', 'rb') as f:
    model10 = torch.load(f)

paths_list = [data_dir + train_filename, data_dir + test_filename]
word_dict, pos_dict = init_vocab_freq(paths_list)
comp_dataset = ParserDataset(word_dict, pos_dict, data_dir, comp_filename, padding=False)
comp_dataloader = DataLoader(comp_dataset)
predictions = predict_data(model10, comp_dataloader)


model_name = 'model10'
loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                       data_dir=data_dir,
                                                       filenames=[train_filename, test_filename],
                                                       word_embedding_size=100,
                                                       pos_embedding_size=25,
                                                       mlp_hidden_dim=100,
                                                       lstm_hidden_layers=3,
                                                       encoder_hidden_size=125,
                                                       alpha=0.25,
                                                       word_embeddings=None,
                                                       epochs=10,
                                                       lr=0.01,
                                                       batch_size=50,
                                                       CUDA=True,
                                                       print_epochs=True,
                                                       save_dir='./dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name, '\tbest train accuracy: ', max(train_acc_list), '\tbest test accuracy: ', max(test_acc_list))


model_name = 'model11'
loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                       data_dir=data_dir,
                                                       filenames=[train_filename, test_filename],
                                                       word_embedding_size=100,
                                                       pos_embedding_size=25,
                                                       mlp_hidden_dim=100,
                                                       lstm_hidden_layers=4,  # changed
                                                       encoder_hidden_size=125,
                                                       alpha=0.25,
                                                       word_embeddings=None,
                                                       epochs=10,
                                                       lr=0.01,
                                                       batch_size=50,
                                                       CUDA=True,
                                                       print_epochs=True,
                                                       save_dir='./dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name, '\tbest train accuracy: ', max(train_acc_list), '\tbest test accuracy: ', max(test_acc_list))


model_name = 'model12'
loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                       data_dir=data_dir,
                                                       filenames=[train_filename, test_filename],
                                                       word_embedding_size=100,
                                                       pos_embedding_size=25,
                                                       mlp_hidden_dim=100,
                                                       lstm_hidden_layers=2,  # changed
                                                       encoder_hidden_size=125,
                                                       alpha=0.25,
                                                       word_embeddings=None,
                                                       epochs=10,
                                                       lr=0.01,
                                                       batch_size=50,
                                                       CUDA=True,
                                                       print_epochs=True,
                                                       save_dir='./dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name, '\tbest train accuracy: ', max(train_acc_list), '\tbest test accuracy: ', max(test_acc_list))


model_name = 'model13'
loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                       data_dir=data_dir,
                                                       filenames=[train_filename, test_filename],
                                                       word_embedding_size=300,  # changed
                                                       pos_embedding_size=25,
                                                       mlp_hidden_dim=100,
                                                       lstm_hidden_layers=3,  # changed
                                                       encoder_hidden_size=125,
                                                       alpha=0.25,
                                                       word_embeddings=None,
                                                       epochs=10,
                                                       lr=0.01,
                                                       batch_size=50,
                                                       CUDA=True,
                                                       print_epochs=True,
                                                       save_dir='./dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name, '\tbest train accuracy: ', max(train_acc_list), '\tbest test accuracy: ', max(test_acc_list))