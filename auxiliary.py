from torch import nn


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


def save_predictions(predictions: list, source_file_path: str, predictions_dir: str, predictions_file_name: str):
    """"""
    predictions_flat = [item for sublist in predictions for item in sublist]  # flatten list
    predictions_filtered = filter(lambda x: x != -1, predictions_flat)  # remove -1 as we do not want to print them
    with open(source_file_path, 'r') as source_file:
        with open(predictions_dir + predictions_file_name, 'w') as prediction_file:
            print_list = []
            pred_iter = iter(predictions_filtered)
            for line in source_file:
                if line == '\n':
                    # print_list.append(str())
                    prediction_file.write('\n')
                    continue
                try:
                    head_pred = next(pred_iter)
                except StopIteration:
                    print('The size of the predictions did not match the size of the source file')
                    break
                split_line = line.split('\t')
                split_line[6] = str(head_pred)
                prediction_file.write('\t'.join(split_line))


def NLLLoss(score_matrix, true_tree_arcs):
    log_softmax = nn.LogSoftmax(dim=0)
    prob_score_matrix = log_softmax(score_matrix)
    size_Y = len(true_tree_arcs) - 1
    log_softmax_sum = 0
    for h, m in true_tree_arcs:
        if h == -1:  # the first arc is fictive
            continue
        log_softmax_sum += prob_score_matrix[h, m]
    return (-1 / size_Y) * log_softmax_sum
