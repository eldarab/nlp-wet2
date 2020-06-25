import torch
import numpy as np
from model import KiperwasserDependencyParser
from torch import optim
from auxiliary import convert_tree_to_list
from data import ParserDataset
from data import init_vocab_freq, init_train_freq
from torch.utils.data.dataloader import DataLoader


# data paths
data_dir = './data/'
train_path = data_dir + 'train.labeled'
test_path = data_dir + 'test.labeled'
comp_path = data_dir + 'comp.unlabeled'

# converting raw data to dedicated data objects
paths_list = [train_path, test_path]
word_dict, pos_dict = init_vocab_freq(paths_list)  # TODO https://moodle.technion.ac.il/mod/forum/discuss.php?d=522050
# train_word_dict = init_train_freq([train_path])

train_dataset = ParserDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train_dataset, shuffle=True)  # batch size is 1 by default
test_dataset = ParserDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
test_dataloader = DataLoader(test_dataset, shuffle=False)

word_vocab_size = len(train_dataset.word_idx_mappings)  # includes words from test
pos_vocab_size = len(train_dataset.pos_idx_mappings)  # includes POSs from test

# setting model hyper-parameters, in the order of appearance in the paper
word_embedding_size = 100  # Word embedding dimension
pos_embedding_size = 25    # POS tag embedding dimension
mlp_hidden_dim = 100       # Hidden units in MLP
lstm_hidden_layers = 2     # BI-LSTM Layers
encoder_hidden_size = 125  # BI-LSTM Dimensions (hidden/output)
alpha = 0.25               # alpha (for word dropout)

epochs = 1
lr = 0.01
accumulate_grad_steps = 50  # effective batch_size
word_embeddings = train_dataset.word_vectors

# creating model
model = KiperwasserDependencyParser(lstm_hidden_layers=lstm_hidden_layers,
                                    word_vocab_size=word_vocab_size,
                                    word_embedding_size=word_embedding_size,
                                    pos_vocab_size=pos_vocab_size,
                                    pos_embedding_size=pos_embedding_size,
                                    encoder_hidden_size=encoder_hidden_size,
                                    mlp_hidden_dim=mlp_hidden_dim)

# moving to CUDA
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
if cuda_available:
    model.cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


def UAS(true_tree_arcs, pred_tree_arcs):
    """
    Calculates UAS (unlabeled accuracy score) for two trees.
    :param true_tree_arcs: A list of tree arcs of the format true_tree_arcs[modifier] = head
    :param pred_tree_arcs: A list of tree arcs of the format pred_tree_arcs[modifier] = head
    :return: The percentage of correct arcs
    """
    num_deps = len(true_tree_arcs)  # number of dependencies in the true tree
    correct = 0
    assert true_tree_arcs[0] == -1 and pred_tree_arcs[0] == -1
    for true, pred in zip(true_tree_arcs, pred_tree_arcs):
        if true == -1:  # fictive arc doesn't count
            continue
        if true == pred:
            correct += 1
    return correct / num_deps


# Training start
print("Training Started")
accuracy_list = []


loss_list = []


for epoch in range(epochs):
    acc = 0
    printable_loss = 0

    for batch_idx, input_data in enumerate(train_dataloader):
        loss, predicted_tree = model(input_data)
        loss = loss / accumulate_grad_steps  # TODO why we do this?
        loss.backward()

        if batch_idx % accumulate_grad_steps == 0:
            optimizer.step()
            model.zero_grad()

        printable_loss += loss.item()

        # calculating accuracy
        _, _, _, true_tree = input_data
        true_tree = convert_tree_to_list(true_tree)
        predicted_tree = list(predicted_tree)
        acc += UAS(predicted_tree, true_tree)

    printable_loss = printable_loss / len(train_dataset)
    acc = acc / len(train_dataset)
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
