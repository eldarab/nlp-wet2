from auxiliary import split
from auxiliary import add_or_append
import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter, defaultdict

# These are not relevant for our POS tagger but might be useful for HW2
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN  # this can be used if you are not padding your batches
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]  # did not add ROOT_TOKEN to here because it's already in the sentence

ROOT_TOKEN_COUNTER = 0
ROOT_TOKEN_HEAD = -1
# TODO consider lowering all words


class ParserDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = [(ROOT_TOKEN_COUNTER, ROOT_TOKEN, ROOT_TOKEN, ROOT_TOKEN_HEAD)]
            for line in f:
                if line == '\n':
                    self.sentences.append(cur_sentence)
                    cur_sentence = [(ROOT_TOKEN_COUNTER, ROOT_TOKEN, ROOT_TOKEN, ROOT_TOKEN_HEAD)]
                    continue
                line_splitted = line.split('\t')
                assert len(line_splitted) >= 6
                token_counter = int(line_splitted[0])
                token = line_splitted[1]
                token_pos = line_splitted[3]
                token_head = int(line_splitted[6])
                cur_sentence.append((token_counter, token, token_pos, token_head))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class ParserDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str,  # TODO why this gets word_dict, pos_dict
                 padding=False, word_embeddings=None, alpha=0.25, train_word_freq=None):
        super().__init__()
        self.alpha = alpha
        self.train_word_freq = train_word_freq

        assert 'train' in subset or 'test' in subset
        self.subset = subset  # One of the following: [train, test]
        self.file = dir_path + subset  # TODO removed .labeled from here
        self.datareader = ParserDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)
        # self.pos_size = len(self.datareader.pos_dict)  # TODO do we need this?
        if word_embeddings:  # TODO receive vocab vectors name
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
            self.word_vector_dim = self.word_vectors.size(-1)
        else:
            self.word_idx_mappings, self.idx_word_mappings = self.init_vocab(self.datareader.word_dict)
            self.word_vectors = None
            self.word_vector_dim = None

        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)
        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len, true_tree_heads

    @staticmethod
    def init_vocab(word_dict):
        vocab = Vocab(Counter(word_dict), specials=SPECIAL_TOKENS)
        return vocab.stoi, vocab.itos

    @staticmethod
    def init_glove_word_embeddings(word_dict):
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors  # For some reason, the indexes are reversed

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):  # TODO still not using padding
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        sentence_true_heads_list = list()

        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            true_tree_heads = []
            for modifier_idx, word, pos, head_idx in sentence:
                # TODO reconsider enabling dropout on root
                if self.subset == 'train' and self.train_word_freq is not None and self.dropout(word):
                    word = UNKNOWN_TOKEN
                    pos = UNKNOWN_TOKEN

                if word not in self.word_idx_mappings:  # TODO what happens if word has no mapping?
                    words_idx_list.append(self.word_idx_mappings.get(UNKNOWN_TOKEN))
                else:
                    words_idx_list.append(self.word_idx_mappings.get(word))

                if pos not in self.pos_idx_mappings:  # TODO what happens if POS has no mapping?
                    pos_idx_list.append(self.pos_idx_mappings.get(UNKNOWN_TOKEN))
                else:
                    pos_idx_list.append(self.pos_idx_mappings.get(pos))

                true_tree_heads.append((head_idx, modifier_idx))
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)
            sentence_true_heads_list.append(true_tree_heads)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list,
                                                                     sentence_true_heads_list))}

    def dropout(self, word):
        drop_prob = self.alpha / (self.alpha+self.train_word_freq[word])
        return torch.bernoulli(torch.tensor(drop_prob))


def init_vocab_freq(list_of_paths):
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == '\n':
                    word_dict[ROOT_TOKEN] += 1
                    pos_dict[ROOT_TOKEN] += 1
                    continue
                line_splitted = line.split('\t')
                assert len(line_splitted) >= 6
                word = line_splitted[1]
                pos = line_splitted[3]
                word_dict[word] += 1
                pos_dict[pos] += 1
    return word_dict, pos_dict


def init_train_freq(list_of_paths):
    word_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                if line == '\n':
                    word_dict[ROOT_TOKEN] += 1
                    continue
                word = line.split('\t')[1]
                word_dict[word] += 1
    return word_dict
