import torch

from train import train_model, draw_graphs

data_dir = 'data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
combined_filename = 'combined.labeled'
train_200_filename = 'train_200.labeled'
test_200_filename = 'test_200.labeled'
comp_filename = 'comp.unlabeled'

model_name = 'model-comp10'
_, loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                          data_dir=data_dir,
                                                          filenames=[combined_filename, test_filename],
                                                          word_embedding_size=100,
                                                          pos_embedding_size=25,
                                                          mlp_hidden_dim=100,
                                                          lstm_hidden_layers=3,
                                                          encoder_hidden_size=125,
                                                          alpha=0.25,
                                                          word_embeddings='glove.840B.300d',
                                                          lowercase=False,
                                                          epochs=10,
                                                          lr=0.01,
                                                          batch_size=50,
                                                          CUDA=True,
                                                          print_epochs=True,
                                                          save_dir='dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print('finished training ', model_name)


model_name = 'model-comp12'
_, loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                          data_dir=data_dir,
                                                          filenames=[combined_filename, test_filename],
                                                          word_embedding_size=100,
                                                          pos_embedding_size=25,
                                                          mlp_hidden_dim=100,
                                                          lstm_hidden_layers=3,
                                                          encoder_hidden_size=125,
                                                          alpha=0.25,
                                                          word_embeddings=None,
                                                          lowercase=False,
                                                          epochs=10,
                                                          lr=0.01,
                                                          batch_size=50,
                                                          CUDA=True,
                                                          print_epochs=True,
                                                          save_dir='dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print('finished training ', model_name)


model_name = 'model-comp11'
_, loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                          data_dir=data_dir,
                                                          filenames=[combined_filename, test_filename],
                                                          word_embedding_size=None,
                                                          pos_embedding_size=25,
                                                          mlp_hidden_dim=100,
                                                          lstm_hidden_layers=3,
                                                          encoder_hidden_size=125,
                                                          alpha=0.25,
                                                          word_embeddings='glove.840B.300d',
                                                          lowercase=False,
                                                          epochs=10,
                                                          lr=0.01,
                                                          batch_size=50,
                                                          CUDA=True,
                                                          print_epochs=True,
                                                          save_dir='dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print('finished training ', model_name)
