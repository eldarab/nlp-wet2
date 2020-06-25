from torch import optim
from torch.utils.data.dataloader import DataLoader
from auxiliary import convert_tree_to_list
from eval import UAS, evaluate
from data import init_vocab_freq, ParserDataset
import numpy as np
from model import KiperwasserDependencyParser
import torch
import matplotlib.pyplot as plt


def train(epochs, batch_size, optimizer, train_dataset, train_dataloader, test_dataset, model, print_epochs=True):
    if print_epochs:
        print("Training Started")

    train_acc_list, test_acc_list, loss_list = [], [], []
    for epoch in range(epochs):
        train_acc = 0
        epoch_loss = 0

        for batch_idx, input_data in enumerate(train_dataloader):
            loss, predicted_tree = model(input_data)
            epoch_loss += loss.item()
            loss = loss / batch_size
            loss.backward()

            if batch_idx % batch_size == 0:
                optimizer.step()
                model.zero_grad()

            # calculating train accuracy
            _, _, _, true_tree = input_data
            true_tree = convert_tree_to_list(true_tree)
            predicted_tree = list(predicted_tree)
            train_acc += UAS(predicted_tree, true_tree)

        epoch_loss = epoch_loss / len(train_dataset)
        train_acc = train_acc / len(train_dataset)
        loss_list.append(float(epoch_loss))
        train_acc_list.append(float(train_acc))
        e_interval = len(train_dataset)
        test_acc = evaluate(model, test_dataset)
        test_acc_list.append(test_acc)
        if print_epochs:
            print("Epoch: {}\tLoss: {}\tTrain Accuracy: {}\tTest Accuracy: {}".
                  format(epoch + 1,
                         np.mean(loss_list[-e_interval:]),
                         np.mean(train_acc_list[-e_interval:]),
                         test_acc))

        return loss_list, train_acc_list, test_acc_list


def train_model(model_name, paths_list, word_embedding_size=100, pos_embedding_size=25, mlp_hidden_dim=100,
                lstm_hidden_layers=2, encoder_hidden_size=125, alpha=0.25, epochs=10, lr=0.1, batch_size=50,
                CUDA=True, print_epochs=True):
    # converting raw data to dedicated data objects
    word_dict, pos_dict = init_vocab_freq(paths_list)  # TODO https://moodle.technion.ac.il/mod/forum/discuss.php?d=522050

    train_dataset = ParserDataset(word_dict, pos_dict, data_dir, 'train_300', padding=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True)  # batch size is 1 by default
    test_dataset = ParserDataset(word_dict, pos_dict, data_dir, 'test_300', padding=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    word_vocab_size = len(train_dataset.word_idx_mappings)  # includes words from test
    pos_vocab_size = len(train_dataset.pos_idx_mappings)  # includes POSs from test

    word_embeddings = train_dataset.word_vectors

    model = KiperwasserDependencyParser(lstm_hidden_layers=lstm_hidden_layers,
                                        word_vocab_size=word_vocab_size,
                                        word_embedding_size=word_embedding_size,
                                        pos_vocab_size=pos_vocab_size,
                                        pos_embedding_size=pos_embedding_size,
                                        encoder_hidden_size=encoder_hidden_size,
                                        mlp_hidden_dim=mlp_hidden_dim)

    if CUDA:
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda_available else "cpu")
        if cuda_available:
            model.cuda()
        else:
            raise Exception('You requested to use CUDA but CUDA is not available')

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_list, train_acc_list, test_acc_list = train(epochs, batch_size, optimizer, train_dataset, train_dataloader,
                                                     test_dataset, model, print_epochs)

    return loss_list, train_acc_list, test_acc_list


def draw_graphs(loss_list, train_acc_list, test_acc_list):
    plt.plot(train_acc_list, c="red", label="Train accuracy")
    plt.plot(test_acc_list, c="green", label="Test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy %")
    plt.legend()
    plt.show()

    plt.plot(loss_list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
