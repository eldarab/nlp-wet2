from data import ParserDataset
from data import init_vocab_freq, init_train_freq
from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import KiperwasserDependencyParser
from torch import optim
import torch


data_dir = './data/'
train_path = data_dir + 'train.labeled'
test_path = data_dir + 'test.labeled'
comp_path = data_dir + 'comp.unlabeled'

paths_list = [train_path, test_path]
word_dict, pos_dict = init_vocab_freq(paths_list)  # TODO https://moodle.technion.ac.il/mod/forum/discuss.php?d=522050
# train_word_dict = init_train_freq([train_path])

train = ParserDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train, shuffle=True)  # batch size is 1 by default
test = ParserDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
test_dataloader = DataLoader(test, shuffle=False)

epochs = 1
lstm_hidden_layers = 2
word_vocab_size = len(train.word_idx_mappings)
word_embedding_size = 100
pos_vocab_size = len(train.pos_idx_mappings)
pos_embedding_size = 25
encoder_hidden_size = 300
mlp_hidden_dim = 100
word_embeddings = None

model = KiperwasserDependencyParser(lstm_hidden_layers=lstm_hidden_layers,
                                    word_vocab_size=word_vocab_size,
                                    word_embedding_size=word_embedding_size,
                                    pos_vocab_size=pos_vocab_size,
                                    pos_embedding_size=pos_embedding_size,
                                    encoder_hidden_size=encoder_hidden_size,
                                    mlp_hidden_dim=mlp_hidden_dim)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

if cuda_available:
    model.cuda()

# We will be using a simple SGD optimizer to minimize the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
accumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

# Training start
print("Training Started")
accuracy_list = []
loss_list = []
for epoch in range(epochs):
    acc = 0
    printable_loss = 0

    for batch_idx, input_data in enumerate(train_dataloader):
        loss, predicted_tree = model(input_data)
        loss = loss / accumulate_grad_steps
        loss.backward()

        if batch_idx % accumulate_grad_steps == 0:
            optimizer.step()
            model.zero_grad()

        printable_loss += loss.item()

    printable_loss = printable_loss / len(train)
    acc = acc / len(train)
    loss_list.append(float(printable_loss))
    accuracy_list.append(float(acc))
    e_interval = len(train_dataloader)
    print(f'Epoch {epoch+1} completted, \t Loss{np.mean(loss_list[-e_interval:])}')
    # test_acc = evaluate()
    # test_acc = 1
    # e_interval = batch_idx
    # print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
    #                                                                               np.mean(loss_list[-e_interval:]),
    #                                                                               np.mean(accuracy_list[-e_interval:]),
    #                                                                               test_acc))
