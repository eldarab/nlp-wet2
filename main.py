from train import train_model2, draw_graphs

data_dir = 'data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
comp_filename = 'comp.unlabeled'

model_name = 'model20'
mode, loss_list, train_acc_list, test_acc_list = train_model2(model_name=model_name,
                                                              data_dir=data_dir,
                                                              filenames=[train_filename, test_filename],
                                                              word_embedding_size=100,
                                                              pos_embedding_size=25,
                                                              mlp_hidden_dim=100,
                                                              lstm_hidden_layers=3,
                                                              encoder_hidden_size=125,
                                                              alpha=0.25,
                                                              word_embeddings=None,
                                                              lowercase=True,  # changed
                                                              epochs=10,
                                                              lr=0.01,
                                                              batch_size=50,
                                                              CUDA=True,
                                                              print_epochs=True,
                                                              save_dir='dumps/')

draw_graphs(loss_list, train_acc_list, test_acc_list, save_path='./dumps/' + model_name + '_graphs.pkl')
print(model_name, '\tbest train accuracy: ', max(train_acc_list), '\tbest test accuracy: ', max(test_acc_list))

