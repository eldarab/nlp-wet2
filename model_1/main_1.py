import torch
from torch.utils.data.dataloader import DataLoader

from eval import predict_data
from model_1.data_1 import ParserDataset, init_vocab_freq
from model_1.train_1 import train_model, draw_graphs


data_dir = '../data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
comp_filename = 'comp.unlabeled'
train_300_filename = 'train_300.labeled'
test_300_filename = 'test_300.labeled'


with open('../dumps/model10/epoch5.eh', 'rb') as f:
    model10 = torch.load(f)

paths_list = [data_dir + train_filename, data_dir + test_filename]
word_dict, pos_dict = init_vocab_freq(paths_list)
comp_dataset = ParserDataset(word_dict, pos_dict, data_dir, comp_filename, padding=False)
comp_dataloader = DataLoader(comp_dataset)
predictions = predict_data(model10, comp_dataloader)


model_name = 'model100'
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
                                                       save_dir='../dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name, '\tbest train accuracy: ', max(train_acc_list), '\tbest test accuracy: ', max(test_acc_list))
