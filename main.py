from train import train_model, draw_graphs


data_dir = './data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
comp_filename = 'comp.unlabeled'
train_300_filename = 'train_300.labeled'
test_300_filename = 'test_300.labeled'

loss_list, train_acc_list, test_acc_list = train_model(model_name='Eldar-Sinai',
                                                       data_dir=data_dir,
                                                       filenames=[train_300_filename, test_300_filename],
                                                       word_embedding_size=100,
                                                       pos_embedding_size=25,
                                                       mlp_hidden_dim=100,
                                                       lstm_hidden_layers=2,
                                                       encoder_hidden_size=125,
                                                       alpha=0.25,
                                                       word_embeddings=None,
                                                       epochs=2,
                                                       lr=0.01,
                                                       batch_size=2,
                                                       CUDA=True,
                                                       print_epochs=True,
                                                       save_dir='./dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list)
