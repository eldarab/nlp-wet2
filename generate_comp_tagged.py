import torch
from torch.utils.data.dataloader import DataLoader
from eval import evaluate, evaluate_old
from data import init_vocab_freq, ParserDataset
from auxiliary import save_predictions
from eval import predict_data

data_dir = 'data/'
train_filename = 'train.labeled'
test_filename = 'test.labeled'
comp_filename = 'comp.unlabeled'


# Generate a tagged comp file using the base model
paths_list = [data_dir + train_filename, data_dir + test_filename]
word_dict, pos_dict = init_vocab_freq(paths_list, lowercase=False)
comp_dataset = ParserDataset(word_dict, pos_dict, data_dir, comp_filename, lowercase=False)
comp_dataloader = DataLoader(comp_dataset, shuffle=False)

with open('trained models/base_model', 'rb') as f:
    advanced_model = torch.load(f)
if torch.cuda.is_available():
    advanced_model.to('cuda')
predictions = predict_data(advanced_model, comp_dataloader)
save_predictions(predictions, 'comp.unlabeled', '', 'comp_m1_318792827.labeled')


# Generate a tagged comp file using the advanced model
paths_list = [data_dir + train_filename, data_dir + test_filename]
word_dict, pos_dict = init_vocab_freq(paths_list, lowercase=True)
comp_dataset = ParserDataset(word_dict, pos_dict, data_dir, comp_filename, lowercase=True)
comp_dataloader = DataLoader(comp_dataset, shuffle=False)

with open('trained models/advanced_model.eh', 'rb') as f:
    advanced_model = torch.load(f)
if torch.cuda.is_available():
    advanced_model.to('cuda')
predictions = predict_data(advanced_model, comp_dataloader)
save_predictions(predictions, data_dir+comp_filename, '', 'comp_m2_318792827.labeled')
