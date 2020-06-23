from auxiliary import split
from auxiliary import add_or_append
import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter

# These are not relevant for our POS tagger but might be useful for HW2
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN  # this can be used if you are not padding your batches
ROOT_TOKEN = "<root>"  # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]  # TODO if we decide to have a special root token, than add it here


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
            cur_sentence = []
            for line in f:
                if line == '\n':
                    self.sentences.append(cur_sentence)
                    cur_sentence = []
                    continue
                line_splitted = line.split('\t')
                assert len(line_splitted) >= 6
                token_counter = line_splitted[0]
                token = line_splitted[1]
                token_pos = line_splitted[3]
                token_head = line_splitted[6]
                cur_sentence.append((token_counter, token, token_pos, token_head))
        pass

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class ParserDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str,  # TODO why this gets word_dict, pos_dict
                 padding=False, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = dir_path + subset + ".labeled"  # TODO in HW2 changed to .labeled or unlabeled
        self.datareader = ParserDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)
        # self.pos_size = len(self.datareader.pos_dict)  # TODO do we need this?
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.datareader.word_dict)
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len

    @staticmethod
    def init_word_embeddings(word_dict):
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
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
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


def generate_dicts(file_list):
    """
    Extracts words and tags vocabularies.
    :return: word2idx, tag2idx, idx2word, idx2tag
    """
    word2idx_dict = {}
    pos2idx_dict = {}
    for file in file_list:
        with open(file, 'r') as f:
            word_counter, pos_counter = 0, 0
            for line in f:  # each line in the test data corresponds to at most one word
                if line == '\n':
                    continue
                line_splitted = line.split('\t')
                assert len(line_splitted) >= 6
                word = line_splitted[1]
                pos = line_splitted[3]
                if word not in word2idx_dict:
                    word2idx_dict[word] = word_counter
                    word_counter += 1
                if pos not in pos2idx_dict:
                    pos2idx_dict[pos] = pos_counter
                    pos_counter += 1

    return word2idx_dict, pos2idx_dict