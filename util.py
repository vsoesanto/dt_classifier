import math


def process_input(all_labels, number_all_docs, all_features):
    training_data = "train.vectors.txt"
    with open(training_data, "r") as td:
        doc_idx = 0
        for line in td:
            line = line.split(" ")
            c = line[0].strip()  # class of this doc
            # print(c)

            # initialize a list for every class in number_all_docs
            if c not in number_all_docs:
                number_all_docs[c] = list()

            if c not in all_labels:
                all_labels.append(c)

            # take inventory of features in this doc
            # this_doc_features = dict()  # key=word, value=occurrence
            this_doc_features = list()
            for i in range(1, len(line), 1):
                word = line[i].split(":")[0]

                if word == "\n":
                    # print("found newline at " + str(line))
                    continue

                # save all the documents belonging to this class
                this_doc_features.append(word)

                # increment the number of docs that have this feature
                if word not in all_features:
                    all_features[word] = dict()
                if c not in all_features[word]:
                    all_features[word][c] = 0
                all_features[word][c] = all_features[word][c] + 1

            number_all_docs[c].append(this_doc_features)
            doc_idx += 1

    return all_labels, number_all_docs, all_features


def compute_info_gain(features, all_labels, number_docs, of):
    best_feature = ""
    max_info_gain = 0
    # min_ent = float("inf")
    for feature in features:
        s_a = 0  # total number of docs that contain feature a
        s = 0  # total number of docs
        post_split_1 = dict()
        post_split_0 = dict()

        for label in all_labels:
            # initialize post_splits: number of documents in this label that contain/don't contain feature
            post_split_1[label] = 0
            post_split_0[label] = 0

            # compute total number of docs
            s += len(number_docs[label])

            # compute total number of docs belonging to [label] containing [feature]
            if label in features[feature]:
                s_a += features[feature][label]

            # count_1 reflects the number of features found in this class
            if label in features[feature]:
                post_split_1[label] = features[feature][label]
                total = len(number_docs[label])
                post_split_0[label] = total - post_split_1[label]

                # if post_split_0[label] < 0:
                    # print("feature=" + feature + " label=" + label + " " + str(post_split_0[label]))

            # count_0 reflects the number of features not found in this class
            if label not in features[feature]:
                post_split_1[label] = 0
                post_split_0[label] = len(number_docs[label])

        # info gain calculation
        h_of_s_feature_1 = compute_avg_entropy(post_split_1, s_a, s)
        h_of_s_feature_0 = compute_avg_entropy(post_split_0, (s - s_a), s)
        avg_ent = h_of_s_feature_1 + h_of_s_feature_0

        # compute h(s)
        info_gain = compute_entropy(number_docs) - avg_ent
        # print("feature=" + feature + ", info_gain=" + str(info_gain))

        if max_info_gain < info_gain:
            max_info_gain = info_gain
            best_feature = feature

    # print("final min=" + str(min_ent) + ", best feature=" + best_feature)
    # of.write("final min=" + str(min_ent) + ", best feature=" + best_feature + "\n")
    # print("final max info gain=" + str(max_info_gain) + ", best feature=" + best_feature)
    of.write("final max info gain=" + str(max_info_gain) + ", best feature=" + best_feature + "\n")
    of.write("\n")
    # print()
    return best_feature, max_info_gain


def compute_entropy(number_docs):
    h_s = 0
    total_docs = 0
    no_docs_by_label = dict()
    for label in number_docs:
        docs_in_label = len(number_docs[label])
        no_docs_by_label[label] = docs_in_label
        total_docs += docs_in_label

    for label in no_docs_by_label:
        prob = no_docs_by_label[label] / total_docs
        # print("prob=" + str(prob))
        if prob == 0.0:
            log_prob = 0.0
        else:
            log_prob = math.log(prob, 2)
        h_s = h_s + (prob * log_prob)

    # print("final h_s=" + str(h_s))
    return abs(h_s)


def compute_avg_entropy(post_split_1, s_a, s):
    h_of_s = 0
    for label in post_split_1:
        # print("computing " + str(post_split_1) )
        # print("s_a=" + str(s_a))
        # print("label=" + label + ", post_split[label]=" + str(post_split_1[label]))
        if s_a == 0:
            prob_label = 0.0
        else:
            prob_label = post_split_1[label] / s_a

        if prob_label == 0.0:
            log_prob = 0
        else:
            # print("prob_label=" + str(prob_label))
            log_prob = math.log(prob_label, 2)

        h_of_s = h_of_s + (prob_label * log_prob)
    h_of_s = abs(h_of_s) * (s_a / s)
    # print(abs(h_of_s_feature_1) * (s_a / s))

    return h_of_s