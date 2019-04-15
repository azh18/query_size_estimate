from sklearn.svm import SVR
import math
import numpy as np
from traditional.util import unnormalize_labels, normalize_labels


def get_q_error(real_labels, pred_labels):
    errors = []
    for i in range(len(real_labels)):
        if pred_labels[i] > real_labels[i]:
            err_unit = math.fabs(float(pred_labels[i])/real_labels[i])
        else:
            err_unit = math.fabs(float(real_labels[i])/pred_labels[i])

        errors.append(err_unit)
    errors.sort()
    median = errors[int(len(errors)*0.5)-1]
    p90 = errors[int(len(errors)*0.9)-1]
    p95 = errors[int(len(errors)*0.95)-1]
    p99 = errors[int(len(errors)*0.99)-1]

    print("median=", median)
    print("p90=", p90)
    print("p95=", p95)
    print("p99=", p99)


def build_SVR_dataset(joins_enc, predicates_enc, label, num_queries, column_min_max_vals):
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    train_data = []
    test_data = []
    for i in range(len(predicates_train)):
        train_data.append(np.hstack([predicates_train[i], joins_train[i]]))
    for i in range(len(predicates_test)):
        test_data.append(np.hstack([predicates_test[i], joins_test[i]]))

    label_min_max_val = [min_val, max_val]
    return train_data, labels_train, test_data, labels_test, column_min_max_vals, label_min_max_val


class SVRModel:
    def __init__(self):
        self.model = SVR(kernel="rbf", gamma='scale', C=0.1)
        self.train_data, self.train_label, self.test_data, self.test_label = None, None, None, None
        self.origin_label_min_max = None

    def bind_dataset(self, train_data, train_label, test_data, test_label, origin_label_min_max):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.origin_label_min_max = origin_label_min_max

    def train_grid(self, gamma_list, c_list):
        for gamma in gamma_list:
            for c in c_list:
                self.model = SVR(kernel="rbf", gamma=gamma, C=c)
                print("gamma = ", gamma, "C = ", c)
                self.model.fit(self.train_data, self.train_label)
                predict_label = self.model.predict(self.train_data)
                train_predict_label = unnormalize_labels(predict_label, self.origin_label_min_max[0], self.origin_label_min_max[1])
                train_real_label = unnormalize_labels(self.train_label, self.origin_label_min_max[0], self.origin_label_min_max[1])
                print("On Training Set:")
                get_q_error(train_real_label, train_predict_label)
                predict_label = self.model.predict(self.test_data)
                test_predict_label = unnormalize_labels(predict_label, self.origin_label_min_max[0], self.origin_label_min_max[1])
                test_real_label = unnormalize_labels(self.test_label, self.origin_label_min_max[0], self.origin_label_min_max[1])
                print("On Test Set:")
                get_q_error(test_real_label, test_predict_label)
                print("-----")
