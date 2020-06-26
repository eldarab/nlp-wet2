from train import train_model, draw_graphs

data_dir = 'data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
train_300_filename = 'train_300.labeled'
test_300_filename = 'test_300.labeled'
comp_filename = 'comp.unlabeled'

model_name = 'testing'
mode, loss_list, train_acc_list, test_acc_list = train_model(model_name=model_name,
                                                             data_dir=data_dir,
                                                             filenames=[train_300_filename, test_300_filename],
                                                             word_embedding_size=100,
                                                             pos_embedding_size=25,
                                                             mlp_hidden_dim=100,
                                                             lstm_hidden_layers=3,
                                                             encoder_hidden_size=125,
                                                             alpha=0.25,
                                                             word_embeddings=None,
                                                             epochs=10,
                                                             lr=0.01,
                                                             batch_size=2,
                                                             CUDA=True,
                                                             print_epochs=True,
                                                             save_dir='dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name,
      '\tbest train accuracy: ', round(max(train_acc_list), 4),
      '\tbest test accuracy: ', round(max(test_acc_list), 4))

