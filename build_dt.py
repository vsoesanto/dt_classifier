import sys
from dtnode import DTNode
from util import compute_info_gain
from model import DecisionTree

MAX_DEPTH = 50
MIN_GAIN = 0

# for running on command line
# training_data = sys.argv[1]
# test_data = sys.argv[2]
# max_depth = int(sys.argv[3])
# min_gain = float(sys.argv[4])
# model_file = sys.argv[5]
# sys_output = sys.argv[6]
# acc_file = sys.argv[7]


def generate_tree(depth, of, model):
    features = model.all_features
    all_labels = model.all_labels
    number_docs = model.number_all_docs
    final_model = model.final_model
    root = model.decision_tree_root
    # print(features)

    best_feature, max_info_gain = compute_info_gain(features, all_labels, number_docs, of)

    if max_info_gain == 0.0:
        leaf(final_model, root, number_docs, of)
    elif depth < MAX_DEPTH and max_info_gain >= MIN_GAIN:
        root.feature = best_feature
        of.write("this node's feature=" + root.feature + "\n")

        left_curr_path = root.path
        if root.path == "":  # first set of children
            left_curr_path = best_feature
        else:
            left_curr_path = root.path + "&" + best_feature
        root.left = DTNode("", left_curr_path, None, None)

        right_curr_path = root.path
        if root.path == "":  # first set of children
            right_curr_path = "!" + best_feature
        else:
            right_curr_path = root.path + "&!" + best_feature
        root.right = DTNode("", right_curr_path, None, None)

        # print("generating subsets of best_feature=" + best_feature + "!")
        # of.write("generating subsets=" + best_feature + "!" + "\n")

        new_number_docs_0 = dict()
        new_number_docs_1 = dict()
        new_features_0 = dict()
        new_features_1 = dict()
        for label in all_labels:  # [guns, mideast, misc]
            # initialize new_number_docs
            if label not in new_number_docs_0:
                new_number_docs_0[label] = list()
            if label not in new_number_docs_1:
                new_number_docs_1[label] = list()

            # generate subsets of data
            # loop over current number_docs to get that belong to this label
            for doc in number_docs[label]:
                # if this document, belonging to this label, doesn't contain best feature
                if best_feature not in doc:
                    new_number_docs_0[label].append(doc)
                    # take inventory of features from this doc
                    for feature in doc:
                        if feature not in new_features_0:
                            new_features_0[feature] = dict()
                        if label not in new_features_0[feature]:
                            new_features_0[feature][label] = 0
                        new_features_0[feature][label] += 1
                # if this document, belonging to this label, contains best feature
                else:
                    new_number_docs_1[label].append(doc)
                    # take inventory of features from this doc
                    for feature in doc:
                        if feature not in new_features_1:
                            new_features_1[feature] = dict()
                        if label not in new_features_1[feature]:
                            new_features_1[feature][label] = 0
                        new_features_1[feature][label] += 1

        children = [DecisionTree(new_features_1, new_number_docs_1, all_labels, final_model, root.left),
                    DecisionTree(new_features_0, new_number_docs_0, all_labels, final_model, root.right)]
        for i in range(len(children)):
            child = children[i]
            if i == 0:
                # print("generating " + best_feature + "'s root.left")
                of.write("generating " + best_feature + "'s root.left" + "\n")
            else:
                # print("generating " + best_feature + "'s root.right")
                of.write("generating " + best_feature + "'s root.right" + "\n")
            generate_tree(depth + 1, of, child)
    else:
        leaf(final_model, root, number_docs, of)


def leaf(final_model, root, number_docs, of):
    # print("leaf, path=" + root.path)
    of.write("leaf, path=" + root.path + "\n")

    total_docs = 0
    for label in number_docs:
        no_docs = len(number_docs[label])
        total_docs += no_docs

        if root.path not in final_model:
            final_model[root.path] = dict()

        if label not in final_model[root.path]:
            final_model[root.path][label] = no_docs

        of.write(label + "=" + str(len(number_docs[label])) + "\n")
        # print(label + "=" + str(len(number_docs[label])))

    node_label = ""
    max_prob = -float("inf")
    for label in final_model[root.path]:
        prob = final_model[root.path][label] / total_docs
        # print("no docs= " + str(final_model[root.path][label]) + ", prob " + label + " " + str(prob))
        root.distribution[label] = prob
        if max_prob < prob:
            max_prob = prob
            node_label = label

    root.feature = node_label
    root.total_docs = total_docs
    # print("root's label=" + root.feature)
    of.write("root's label=" + root.feature + "\n")

    of.write("\n")
    # print()


def print_tree(root, string):
    if root is not None:
        string += "node feature=" + root.feature + " path=" + root.path + "\n"
        string = print_tree(root.left, string)
        string = print_tree(root.right, string)
    return string



