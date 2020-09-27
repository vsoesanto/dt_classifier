from model import DecisionTree
import build_dt


def report_model(final_model):
    with open("model_file", "w") as mfile:
        for path in final_model:
            # print(path, end=" ")
            mfile.write(path + " ")

            total_docs = 0
            for label in final_model[path]:
                total_docs += final_model[path][label]

            # print(str(total_docs), end=" ")
            mfile.write(str(total_docs) + " ")
            for label in final_model[path]:
                prob = final_model[path][label] / total_docs
                # print(label + " " + str(prob), end=" ")
                mfile.write(label + " " + str(prob) + " ")
            # print()
            mfile.write("\n")


def tree_crawl(doc, root):
    if root.left is None and root.right is None:
        # print("\treached class=" + root.feature)
        return root.distribution
    else:
        if root.feature in doc:
            # print("\tfeature=" + root.feature + " is in this doc")
            return tree_crawl(doc, root.left)
        else:
            # print("\tfeature=" + root.feature + " is not in this doc")
            return tree_crawl(doc, root.right)


def run(run_type, model, outfile, label_indexes):
    all_labels = model.all_labels

    # initialize a confusion matrix
    confusion_matrix = list()
    for i in range(len(all_labels)):
        row = list()
        for j in range(len(all_labels)):
            row.append(0)
        confusion_matrix.append(row)

    if run_type == "test":
        outfile.write("%%%%% test data:" + "\n")
        file_path = "test.vectors.txt"
    else:
        outfile.write("%%%%% train data:" + "\n")
        file_path = "train.vectors.txt"

    data_file = open(file_path, "r")

    no = -1
    for line in data_file:
        no += 1
        line = line.split(" ")
        truth_label = line[0].strip()

        doc = list()
        # process test doc
        outfile.write("array:" + str(no) + " ")
        for i in range(1, len(line), 1):
            word = line[i].split(":")[0]
            if word == "\n":
                continue
            doc.append(word)

        # print("doc" + str(no) + ": " + str(doc))

        # run doc on tree
        dist = tree_crawl(doc, model.decision_tree_root)
        # print("dist=" + str(dist))
        system_label = ""
        max_prob = -float("inf")

        for c in dist:  # looping over distribution returned from leaf node (separated by class)
            prob = dist[c]
            # determine what the system output label is
            # based on majority vote
            if prob > max_prob:
                max_prob = prob
                system_label = c
            # print(c + " " + str(prob) + " ")  # report what this tree classifies this doc as
        #     outfile.write(c + " " + str(prob) + " ")
        # outfile.write("\n")

        # print("system label=" + system_label + ", idx=" + str(label_indexes[system_label]))
        outfile.write("prediction label=" + system_label + " " + "truth label=" + truth_label + "\n")
        # print("truth label=" + truth_label + ", idx=" + str(label_indexes[truth_label]))

        # if final_label == label:
        system_idx = label_indexes[system_label]
        truth_idx = label_indexes[truth_label]
        # print("inserting at i=" + str(truth_idx) + ", j=" + str(system_idx))
        confusion_matrix[truth_idx][system_idx] += 1
        # print("updated confusion matrix!")
        # for i in range(len(confusion_matrix)):
            # print(confusion_matrix[i])

        # print("\n""\n")
    data_file.close()
    report_acc(model, confusion_matrix, "Test")


def report_acc(model, confusion_matrix, type):
    # accf.write("Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n")
    # accf.write("             talk.politics.guns talk.politics.mideast talk.politics.misc\n")

    print("Confusion matrix for the test data:")
    print("row is the truth, column is the system output\n")
    print("             talk.politics.guns talk.politics.mideast talk.politics.misc")


    total_docs = 0
    total_correct = 0
    for i in range(len(confusion_matrix)):
        label = model.all_labels[i]
        # accf.write(label + " ")
        print(label, end=" ")
        for j in range(len(confusion_matrix[i])):
            item = confusion_matrix[i][j]
            # accf.write(str(item) + " ")
            print(str(item), end=" ")

            total_docs += item

            if i == j:
                total_correct += item

        # print(confusion_matrix[i])
        # accf.write("\n")
        print("")
    # accf.write("\n")
    print("")

    # print("total correct=" + str(total_correct))
    acc = total_correct / total_docs

    # accf.write(" " + type + " accuracy=" + str(acc))
    # accf.write("\n\n")
    print(" " + type + " accuracy=" + str(acc))
    print("")


model = DecisionTree()
with open("analysis", 'w') as of:
    build_dt.generate_tree(0, of, model)

# print model
report_model(model.final_model)

label_indexes = dict()
for i in range(len(model.all_labels)):
    label_indexes[model.all_labels[i]] = i

with open("output", "w") as outfile:
    run("train", model, outfile, label_indexes)
    run("test", model, outfile, label_indexes)