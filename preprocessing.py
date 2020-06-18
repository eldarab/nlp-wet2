from auxiliary import split


class ParseDataReader:
    def __init__(self, file):
        self.file = file
        # TODO figure what the frick these are about
        # self.word_dict = word_dict
        # self.pos_dict = pos_dict
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
                split_line = line.split('\t')
                assert len(split_line) >= 6
                token_counter = split_line[0]
                token = split_line[1]
                token_pos = split_line[3]
                token_head = split_line[6]
                cur_sentence.append((token_counter, token, token_pos, token_head))
        pass

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)
