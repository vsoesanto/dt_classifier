from dtnode import DTNode
from util import process_input


class DecisionTree:

    def __init__(self, all_features=None,
                 number_all_docs=None,
                 all_labels=None,
                 final_model=None,
                 decision_tree_root=None):
        new = True

        if all_features is None:
            self.all_features = dict()
        else:
            # number of documents that contain feature, by class key=feature, value=dict(class: no docs)
            self.all_features = all_features
            new = False

        if number_all_docs is None:
            self.number_all_docs = dict()
        else:
            # all docs by class, # key=class, value=list docs containting a list of its words
            self.number_all_docs = number_all_docs
            new = False

        if all_labels is None:
            self.all_labels = list()  # a list of labels
        else:
            self.all_labels = all_labels
            new = False

        if final_model is None:
            self.final_model = dict()  # path to class to
        else:
            self.final_model = final_model
            new = False

        if decision_tree_root is None:
            self.decision_tree_root = DTNode("", "", None, None)
        else:
            self.decision_tree_root = decision_tree_root
            new = False

        if new:
            self.all_labels, self.number_all_docs, self.all_features = process_input(self.all_labels,
                                                                                    self.number_all_docs,
                                                                                    self.all_features)




