def add_or_append(dictionary, item, size=1):
    """
    Add size to the key item if it is in the dictionary, otherwise appends the key to the dictionary
    """
    if item not in dictionary:
        dictionary[item] = size
    else:
        dictionary[item] += size


def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters: (string) characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def convert_tree_to_list(tree):
    """
    Converts a given tree to the format tree[modifier] = head.
    :param tree: The tree to convert
    :return: A list of the format tree[modifier] = head
    """
    tree = list(tree)
    tree_list = []
    for head, modifier in tree:
        tree_list.append(int(head))
    return tree_list


# def generate_dicts(file_list):
#     """
#     Extracts words and tags vocabularies.
#     :return: word2idx, tag2idx, idx2word, idx2tag
#     """
#     word2idx_dict = {ROOT_TOKEN: 0}
#     pos2idx_dict = {ROOT_TOKEN: 0}
#     for file in file_list:
#         with open(file, 'r') as f:
#             word_counter, pos_counter = 1, 1  # already used 0 for ROOT_TOKEN
#             for line in f:  # each line in the test data corresponds to at most one word
#                 if line == '\n':
#                     continue
#                 line_splitted = line.split('\t')
#                 assert len(line_splitted) >= 6
#                 word = line_splitted[1]
#                 pos = line_splitted[3]
#                 if word not in word2idx_dict:
#                     word2idx_dict[word] = word_counter
#                     word_counter += 1
#                 if pos not in pos2idx_dict:
#                     pos2idx_dict[pos] = pos_counter
#                     pos_counter += 1
#
#     return word2idx_dict, pos2idx_dict
