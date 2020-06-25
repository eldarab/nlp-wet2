from model import KiperwasserDependencyParser


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
