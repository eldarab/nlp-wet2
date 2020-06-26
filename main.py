from train import train_model
from data import ParserDataset, init_vocab_freq
from eval import predict_data
from torch.utils.data.dataloader import DataLoader
from auxiliary import save_predictions


data_dir = './data/'
predictions_dir = 'predictions/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
comp_filename = 'comp.unlabeled'
train_300_filename = 'train_300.labeled'
test_300_filename = 'test_300.labeled'

model, loss_list, train_acc_list, test_acc_list \
    = train_model(model_name='gibberish',
                  data_dir=data_dir,
                  filenames=[train_300_filename, test_300_filename],
                  word_embedding_size=100,
                  pos_embedding_size=25,
                  mlp_hidden_dim=100,
                  lstm_hidden_layers=2,
                  encoder_hidden_size=125,
                  alpha=0.25,
                  word_embeddings=None,
                  epochs=1,
                  lr=0.001,
                  batch_size=20,
                  CUDA=True,
                  print_epochs=True,
                  save_dir='./dumps/')

# draw_graphs(loss_list, train_acc_list, test_acc_list)

word_dict, pos_dict = init_vocab_freq([data_dir+train_300_filename, data_dir+test_300_filename])
comp_dataset = ParserDataset(word_dict, pos_dict, data_dir, comp_filename)
comp_dataloader = DataLoader(comp_dataset)
predictions = predict_data(model, comp_dataloader)
save_predictions(predictions, data_dir+comp_filename, predictions_dir, 'temp')
pass
