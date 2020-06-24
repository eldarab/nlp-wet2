from data import ParserDataset
from data import init_vocab_freq
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
train = ParserDataset(word_dict, pos_dict, data_dir, 'train', padding=False)
train_dataloader = DataLoader(train, shuffle=True)
test = ParserDataset(word_dict, pos_dict, data_dir, 'test', padding=False)
test_dataloader = DataLoader(test, shuffle=False)


epochs = 1
lstm_hidden_layers = 125
word_vocab_size = len(word_dict)
word_embedding_size = 100
pos_vocab_size = len(pos_dict)
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
acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

# Training start
print("Training Started")
accuracy_list = []
loss_list = []
for epoch in range(epochs):
    acc = 0
    printable_loss = 0
    i = 0
    for batch_idx, input_data in enumerate(train_dataloader):
        i += 1

        loss, predicted_tree = model(input_data)
        # tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)
        # print("tag_scores shape -", tag_scores.shape)
        # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
        loss = loss_function(tag_scores, pos_idx_tensor.to(device))
        loss = loss / acumulate_grad_steps
        loss.backward()

        if i % acumulate_grad_steps == 0:
            optimizer.step()
            model.zero_grad()
        printable_loss += loss.item()
        _, indices = torch.max(tag_scores, 1)
        # print("tag_scores shape-", tag_scores.shape)
        # print("indices shape-", indices.shape)
        # acc += indices.eq(pos_idx_tensor.view_as(indices)).mean().item()
        acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
    printable_loss = printable_loss / len(train)
    acc = acc / len(train)
    loss_list.append(float(printable_loss))
    accuracy_list.append(float(acc))
    test_acc = evaluate()
    e_interval = i
    print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
                                                                                  np.mean(loss_list[-e_interval:]),
                                                                                  np.mean(accuracy_list[-e_interval:]),
                                                                                  test_acc))

# debugging
print('=' * 20, 'Debugging', '=' * 20)

DNN = KiperwasserDependencyParser(2, len(word_dict), train.word_vectors.shape[1], len(pos_dict), 25, 100, 100)
sentence0 = train.sentences_dataset[0]
loss, predicted_tree = DNN(sentence0)

test_word_vectors_np = test.word_vectors.numpy()
word_vectors_with_zero_norm = 0
for i, word_vector in enumerate(test_word_vectors_np):
    if np.linalg.norm(word_vector) == 0.0:
        word_vectors_with_zero_norm += 1

print('number of word vectors with zero norm', word_vectors_with_zero_norm)
print('out of ', len(test_word_vectors_np), ' total unique word vectors')
pass
