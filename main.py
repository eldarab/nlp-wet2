from data import ParserDataReader
from data import ParserDataset
from data import generate_dicts
from torch.utils.data.dataloader import DataLoader


data_dir = './data/'
train_path = data_dir + 'train.labeled'
test_path = data_dir + 'test.labeled'
comp_path = data_dir + 'comp.unlabeled'

paths_list = [train_path, test_path]
word_dict, pos_dict = generate_dicts(paths_list)  # TODO https://moodle.technion.ac.il/mod/forum/discuss.php?d=522050
train = ParserDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train, shuffle=True)
test = ParserDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
test_dataloader = DataLoader(test, shuffle=False)

pass
