from train import train_model, draw_graphs


# data paths
data_dir = './data/'
train_path = data_dir + 'train.labeled'
test_path = data_dir + 'test.labeled'
comp_path = data_dir + 'comp.unlabeled'
train_300_path = data_dir + 'train_300.labeled'
test_300_path = data_dir + 'test_300.labeled'

loss_list, train_acc_list, test_acc_list = train_model(model_name='Eldar Sinai',
                                                       paths_list=[train_300_path, test_300_path],
                                                       word_embedding_size=100,
                                                       pos_embedding_size=25,
                                                       mlp_hidden_dim=100,
                                                       lstm_hidden_layers=2,
                                                       encoder_hidden_size=125,
                                                       alpha=0.25,
                                                       epochs=10,
                                                       lr=0.1,
                                                       batch_size=2,
                                                       CUDA=True,
                                                       print_epochs=True)

draw_graphs(loss_list, train_acc_list, test_acc_list)
