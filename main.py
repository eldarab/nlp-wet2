from data import ParserDataReader
from data import ParserDataset
from data import generate_dicts
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import KiperwasserDependencyParser


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

# debugging
print('=' * 20, 'Debugging', '=' * 20)

DNN = KiperwasserDependencyParser(2, len(word_dict), train.word_vectors.shape[1], len(pos_dict), 25, 100, 100)
sentence0 = train.sentences_dataset[0]
output = DNN.forward(sentence0)
output2 = output.detach().numpy()

test_word_vectors_np = test.word_vectors.numpy()
word_vectors_with_zero_norm = 0
for i, word_vector in enumerate(test_word_vectors_np):
    if np.linalg.norm(word_vector) == 0.0:
        word_vectors_with_zero_norm += 1

print('number of word vectors with zero norm', word_vectors_with_zero_norm)
print('out of ', len(test_word_vectors_np), ' total unique word vectors')
pass
