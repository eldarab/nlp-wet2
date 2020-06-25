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
