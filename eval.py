import torch
from torch.utils.data.dataloader import DataLoader
from auxiliary import convert_tree_to_list


def UAS(true_tree_arcs, pred_tree_arcs):
    """
    Calculates UAS (unlabeled accuracy score) for two trees.
    :param true_tree_arcs: A list of tree arcs of the format true_tree_arcs[modifier] = head
    :param pred_tree_arcs: A list of tree arcs of the format pred_tree_arcs[modifier] = head
    :return: The percentage of correct arcs
    """
    num_deps = len(true_tree_arcs) - 1  # number of dependencies in the true tree, minus fictive edge
    correct = 0
    assert true_tree_arcs[0] == -1 and pred_tree_arcs[0] == -1
    for true, pred in zip(true_tree_arcs, pred_tree_arcs):
        if true == -1:  # fictive arc doesn't count
            continue
        if true == pred:
            correct += 1
    return correct / num_deps


def evaluate(model, dataset):
    """
    Gets a trained model and a dataset to check the model's accuracy on
    :param dataset: A dataset object with the data to evaluate the model on
    :param model: Preferably trained Kipperwasser model object
    :return: Accuracy of the model on the given dataset
    """
    dataloader = DataLoader(dataset, shuffle=False)
    accuracy_sum = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            _, _, _, true_tree = input_data
            _, pred_tree = model(input_data)
            true_tree = convert_tree_to_list(true_tree)
            pred_tree = list(pred_tree)
            accuracy_sum += UAS(true_tree, pred_tree)
    return accuracy_sum / len(dataset)


def evaluate_model(model, data_path):
    pass