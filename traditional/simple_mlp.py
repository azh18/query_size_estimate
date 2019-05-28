import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from traditional.data import load_and_encode_train_data
from traditional.util import normalize_labels, unnormalize_labels
import time
import numpy as np


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

# loss without considering unnormalization
def qerror_loss(preds, targets):
    qerror = []
    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


# loss with considering unnormalization
def unnormalized_qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))

# Define model architecture

class SimpleMLP(nn.Module):
    def __init__(self, predicate_feats, join_feats, hid_units):
        super(SimpleMLP, self).__init__()

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 2, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, predicates, joins):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))


        # hid_predicate = hid_predicate * predicate_mask
        # hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        # predicate_norm = predicate_mask.sum(1, keepdim=False)
        # hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        # hid_join = hid_join * join_mask
        # hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        # join_norm = join_mask.sum(1, keepdim=False)
        # hid_join = hid_join / join_norm

        hid = torch.cat((hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out


def build_torch_dataset(joins_enc, predicates_enc, labels, num_queries):
    normed_labels, min_val, max_val = normalize_labels(labels)
    label_min_max_values = [float(min_val), float(max_val)]
    num_trains = 0.9 * num_queries
    end_tests = num_queries
    train_predicates = np.vstack(predicates_enc[:int(num_trains)])
    train_predicates = torch.FloatTensor(train_predicates)
    train_joins = np.vstack(joins_enc[:int(num_trains)])
    train_joins = torch.FloatTensor(train_joins)
    train_labels = np.vstack(normed_labels[:int(num_trains)])
    tensor_train_labels = torch.FloatTensor(train_labels)

    test_predicates = np.vstack(predicates_enc[int(num_trains):int(end_tests)])
    test_predicates = torch.FloatTensor(test_predicates)
    test_joins = np.vstack(joins_enc[int(num_trains):int(end_tests)])
    test_joins = torch.FloatTensor(test_joins)
    test_labels = np.vstack(normed_labels[int(num_trains):int(end_tests)])
    tensor_test_labels = torch.FloatTensor(test_labels)

    train_dataset = TensorDataset(train_predicates, train_joins, tensor_train_labels)
    test_dataset = TensorDataset(test_predicates, test_joins, tensor_test_labels)
    return train_dataset, test_dataset, train_labels, test_labels, label_min_max_values


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        predicates, joins, targets = data_batch

        if cuda:
            predicates, joins, targets = predicates.cuda(), joins.cuda(), targets.cuda()
        predicates, joins, targets = Variable(predicates), Variable(joins), Variable(
            targets)

        t = time.time()
        outputs = model(predicates, joins)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

        return preds, t_total


class SimpleMLPLearner:
    def __init__(self):
        pass

    @staticmethod
    def train_and_predict():
        cuda = False
        num_queries = 50000
        batch_size = 1024
        n_hidden = 256
        predicates_enc, joins_enc, label = load_and_encode_train_data(1000)
        train_dataset, test_dataset, train_labels, test_labels, label_min_max_values = build_torch_dataset(joins_enc, predicates_enc, label, num_queries)

        join_feats = len(joins_enc[0])
        predicates_feats = len(predicates_enc[0])
        # print("join_feats=", join_feats, "predicates_feats=", predicates_feats)
        model = SimpleMLP(predicates_feats, join_feats, n_hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if cuda:
            model.cuda()
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size)


        n_epoch = 1000
        model.train()
        for epoch in range(n_epoch):
            loss_total = 0.0
            for batch_idx, data_batch in enumerate(train_data_loader):
                predicates, joins, labels = data_batch
                if cuda:
                    predicates, joins, labels = predicates.cuda(), joins.cuda(), labels.cuda()
                predicates, joins, labels = Variable(predicates), Variable(joins), Variable(labels)
                optimizer.zero_grad()

                outputs = model(predicates, joins)
                loss = unnormalized_qerror_loss(outputs, labels, label_min_max_values[0], label_min_max_values[1])
                loss_total += loss.item()
                loss.backward()
                optimizer.step()
            print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

            min_val, max_val = label_min_max_values[0], label_min_max_values[1]
            preds_train, t_total = predict(model, train_data_loader, cuda)
            preds_test, t_total = predict(model, test_data_loader, cuda)
            real_preds_train = unnormalize_labels(preds_train, min_val, max_val)
            real_preds_test = unnormalize_labels(preds_test, min_val, max_val)
            real_targets_train = unnormalize_labels(train_labels, min_val, max_val)
            real_targets_test = unnormalize_labels(test_labels, min_val, max_val)
            # Print metrics


            print("\nQ-Error validation set:")
            print_qerror(real_preds_test, real_targets_test)
            # run on validation dataset

        min_val, max_val = label_min_max_values[0], label_min_max_values[1]
        preds_train, t_total = predict(model, train_data_loader, cuda)
        preds_test, t_total = predict(model, test_data_loader, cuda)
        real_preds_train = unnormalize_labels(preds_train, min_val, max_val)
        real_preds_test = unnormalize_labels(preds_test, min_val, max_val)
        real_targets_train = unnormalize_labels(train_labels, min_val, max_val)
        real_targets_test = unnormalize_labels(test_labels, min_val, max_val)
        # Print metrics

        print("\nQ-Error training set:")
        print_qerror(real_preds_train, real_targets_train)

        print("\nQ-Error validation set:")
        print_qerror(real_preds_test, real_targets_test)
