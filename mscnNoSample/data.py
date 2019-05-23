import csv
import torch
from torch.utils.data import dataset
import pickle
import numpy as np

from mscnNoSample.util import *


def make_multiple_slice(n_slice):
    ret = []
    for i in range(n_slice):
        ret.append([])
    return ret


def compute_new_hist(old_ticks, old_hists, new_ticks):
    n_interval = len(new_ticks) - 1
    new_hists = []
    for i in range(n_interval):
        hv_sum = 0
        new_lb = new_ticks[i]
        new_ub = new_ticks[i+1]
        for j in range(n_interval):
            lb = old_ticks[j]
            ub = old_ticks[j+1]
            if lb >= new_ub:
                break
            elif ub <= new_lb:
                continue
            else:
                height = old_hists[j]
                valid_ub = ub if ub < new_ub else new_ub
                valid_lb = lb if lb > new_lb else new_lb
                width = valid_ub - valid_lb
                hv_sum += height * (width / (ub - lb))
        new_hists.append(hv_sum)
    return new_hists


def merge_hist(tick1, tick2, hist1, hist2):
    n_interval = len(hist1)
    lb = min(tick1[0], tick2[0])
    ub = max(tick1[n_interval], tick2[n_interval])
    new_ticks = list(np.array(list(range(n_interval+1)))/n_interval*(ub - lb) + lb)
    new_hist1 = compute_new_hist(tick1, hist1, new_ticks)
    new_hist2 = compute_new_hist(tick2, hist2, new_ticks)
    return new_ticks, new_hist1, new_hist2


class Scalar:
    # if remove_zero is True, replace all numbers less than 1 to 1, avoid negative log result
    def __init__(self, min_val=1, max_val=1, apply_log=False, remove_zero=False):
        self.min_val = min_val
        self.max_val = max_val
        self.apply_log = apply_log
        self.remove_zero = remove_zero
        if self.apply_log:
            if self.remove_zero:
                if self.min_val < 1:
                    self.min_val = 1
                if self.max_val < 1:
                    raise Exception  # should not exist
            self.min_val = np.log(self.min_val)
            self.max_val = np.log(self.max_val)

    def fit(self, data_list):
        self.min_val = min(data_list)
        self.max_val = max(data_list)
        if self.apply_log:
            if self.remove_zero:
                if self.min_val < 1:
                    self.min_val = 1
                if self.max_val < 1:
                    raise Exception  # should not exist
            self.min_val = np.log(self.min_val)
            self.max_val = np.log(self.max_val)

    def transform(self, data_list):
        ret = []
        for d in data_list:
            if self.apply_log:
                if self.remove_zero:
                    if d < 1:
                        d = 1
                d = np.log(d)
            ret.append((d-self.min_val)/(self.max_val-self.min_val))
        return ret

    def inverse_transform(self, data_list):
        ret = []
        for d in data_list:
            val = d * (self.max_val-self.min_val) + self.min_val
            if self.apply_log:
                val = np.exp(val)
            ret.append(val)
        return ret

    def fit_transform(self, data_list):
        self.fit(data_list)
        return self.transform(data_list)


class DataLoader:
    def __init__(self, file_name_train_data, file_name_meta, num_materialized_samples):
        metas = pickle.load(open(file_name_meta, "rb"))
        self.table2abbr = metas[0]
        self.abbr2table = metas[1]
        self.columns = metas[2]
        self.joins = metas[3]
        self.col_min_max = metas[4]
        self.table_abbr_size = metas[5]
        self.raw_histogram = metas[6]
        self.col_n_unique_values = metas[7]
        self.file_name_train_data = file_name_train_data
        self.num_materialized_samples = num_materialized_samples
        # define scalars
        self.join_hist_scalar = Scalar(apply_log=True, remove_zero=True)
        self.table_size_scalar = Scalar(apply_log=True)
        self.col_val_scalar_dict = {}
        self.n_unique_scalar = Scalar(apply_log=True)
        self.ehr_scalar = Scalar(min_val=1e-10, max_val=1, apply_log=True)
        # one-hot meta
        self.column2vec = None
        self.idx2column = None
        self.join2vec = None
        self.idx2join = None
        self.op2vec = None
        self.idx2op = None
        # dim
        self.pred_dim = 0
        self.join_dim = 0
        self.selectivity_dim = 0
        self.histogram_dim = 10

    @staticmethod
    def load_data(file_name, num_materialized_samples, num_queries=None):
        joins = []
        predicates = []
        labels = []
        q_cnt = 0
        # Load queries
        with open(file_name, 'rU') as f:
            data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
            for row in data_raw:
                if row[1] != "":
                    join = row[1].split(',')
                else:
                    join = []
                if row[2] != "":
                    pred = row[2].split(',')
                else:
                    # no predicates
                    pred = []

                if len(pred) > 0:
                    joins.append(join)
                    predicates.append(pred)
                else:
                    # DO NOT ALLOW SQL with no predicates
                    continue

                if int(row[3]) < 1:
                    print("Queries must have non-zero cardinalities")
                    exit(1)
                labels.append(row[3])

                q_cnt += 1
                if q_cnt >= num_queries:
                    break
        print("Loaded queries")

        # Split predicates
        predicates = [list(chunks(d, 3)) for d in predicates]

        return joins, predicates, labels

    def load_n_unique_scalar(self):
        n_uniques = self.col_n_unique_values.values()
        self.n_unique_scalar.fit(n_uniques)
        return self.n_unique_scalar

    def load_table_size_scalar(self):  # (0,max)-scalar
        min_val = 0
        max_val = max(self.table_abbr_size.values())
        self.table_size_scalar = Scalar(min_val=min_val, max_val=max_val, apply_log=True, remove_zero=True)
        return self.table_size_scalar

    def load_col_value_scalar(self, col):
        if col not in self.col_val_scalar_dict:
            self.col_val_scalar_dict[col] = Scalar(min_val=self.col_min_max[col][0], max_val=self.col_min_max[col][1])
        return self.col_val_scalar_dict[col]

    def encode_dataset(self, joins, predicates):
        column2vec = self.column2vec
        join2vec = self.join2vec
        op2vec = self.op2vec
        predicates_enc = []
        joins_enc = []
        join_selectivity_enc = []
        for i, query in enumerate(predicates):
            predicates_enc.append(list())
            joins_enc.append(list())
            join_selectivity_enc.append(list())
            pred_dict = {}  # col->(op, val)
            for predicate in query:
                if len(predicate) == 3:
                    # Proper predicate
                    column = predicate[0]
                    operator = predicate[1]
                    val = int(predicate[2])
                    col_scalar = self.load_col_value_scalar(column)
                    norm_val = col_scalar.transform([val])[0]
                    if column in pred_dict:
                        pred_dict[column].append((operator, norm_val))
                    else:
                        pred_dict[column] = [(operator, norm_val)]
            # deal with <>
            for col in pred_dict.keys():
                if len(pred_dict[col]) == 2:
                    lower_bound = None
                    upper_bound = None
                    for p in pred_dict[col]:
                        if p[0] == "<":
                            upper_bound = p[1]
                        else:
                            lower_bound = p[1]
                    if lower_bound is None or upper_bound is None:
                        raise Exception
                    pred_dict[col] = ("<>", lower_bound, upper_bound)
                elif len(pred_dict[col]) == 1:
                    pred_dict[col] = pred_dict[col][0]
                else:
                    raise Exception
            # encode preds
            for col in pred_dict.keys():
                pred_block = pred_dict[col]
                op = pred_block[0]
                lower_bound = 0
                scale = 0
                tab_abbr = col.split(".")[0]
                table_size = self.table_abbr_size[tab_abbr]
                norm_table_size = self.table_size_scalar.transform([table_size])[0]
                # compute estimate_hit_rate
                norm_ticks = self.col_val_scalar_dict[col].transform(self.raw_histogram[col][0])
                norm_hists = list(np.array(self.raw_histogram[col][1]) / max(self.raw_histogram[col][1]))
                estimate_hit_rate = 0
                if op == "<>":
                    lower_bound = pred_block[1]
                    scale = pred_block[2] - lower_bound
                    beg_idx, end_idx = -1, -1
                    for idx in range(len(self.raw_histogram[col][1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        if lb <= lower_bound < ub:
                            beg_idx = idx
                        if lb <= lower_bound + scale < ub:
                            end_idx = idx
                    for idx in range(beg_idx, end_idx + 1):
                        estimate_hit_rate += norm_hists[idx]
                elif op == "=":
                    lower_bound = pred_block[1]
                    scale = 1
                    for idx in range(len(self.raw_histogram[col][1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        if lb <= lower_bound < ub:
                            estimate_hit_rate += norm_hists[idx]
                            break
                elif op == "<":
                    lower_bound = 0
                    scale = pred_block[1]
                    for idx in range(len(self.raw_histogram[col][1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        estimate_hit_rate += norm_hists[idx]
                        if lb <= lower_bound < ub:
                            break
                else:  # op == >
                    lower_bound = pred_block[1]
                    scale = 1 - lower_bound
                    beg_idx = -1
                    for idx in range(len(self.raw_histogram[col][1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        if lb <= lower_bound < ub:
                            beg_idx = idx
                    estimate_hit_rate += sum(norm_hists[beg_idx:])
                norm_ehr = self.ehr_scalar.transform([estimate_hit_rate])[0]
                pred_vec = []
                pred_vec.append(op2vec[op])
                pred_vec.append(column2vec[col])
                pred_vec.append([lower_bound, scale, norm_ehr, norm_table_size])
                pred_vec = np.hstack(pred_vec)
                self.pred_dim = len(pred_vec)
                predicates_enc[i].append(pred_vec)

            # Join instruction
            for join in joins[i]:
                col1, col2 = join.split("=")
                tab1, tab2 = col1.split(".")[0], col2.split(".")[0]
                col1_n_unique, col2_n_unique, col1_min, col2_min, col1_max, col2_max = \
                    self.col_n_unique_values[col1], self.col_n_unique_values[col2],\
                    self.col_min_max[col1][0], self.col_min_max[col2][0], \
                    self.col_min_max[col1][1], self.col_min_max[col2][1]
                col1_ticks, col1_hist = self.raw_histogram[col1]
                col2_ticks, col2_hist = self.raw_histogram[col2]
                col_min, col_max = min(col1_min, col2_min), max(col1_max, col2_max)
                # merge histogram
                new_ticks, new_hist1, new_hist2 = merge_hist(col1_ticks, col2_ticks, col1_hist, col2_hist)
                # construct join_vec
                norm_tab1_size = self.table_size_scalar.transform([self.table_abbr_size[tab1]])[0]
                norm_tab2_size = self.table_size_scalar.transform([self.table_abbr_size[tab2]])[0]
                norm_hist1 = self.join_hist_scalar.fit_transform([0] + new_hist1)[1:]  # (0, max)-scalar
                norm_hist2 = self.join_hist_scalar.fit_transform([0] + new_hist2)[1:]  # (0, max)-scalar
                norm_n_unique1 = self.n_unique_scalar.transform([col1_n_unique])[0]
                norm_n_unique2 = self.n_unique_scalar.transform([col2_n_unique])[0]
                join_onehot = join2vec[join]
                join_vec = list(join_onehot) + [norm_tab1_size, norm_tab2_size]
                self.join_dim = len(join_vec)
                joins_enc[i].append(join_vec)
                selectivity_vec = [norm_n_unique1, norm_n_unique2] + norm_hist1 + norm_hist2
                self.selectivity_dim = len(selectivity_vec)
                join_selectivity_enc[i].append(selectivity_vec)

            # if no join in this sql, use all-zero encode for the join
            if len(joins[i]) == 0:
                joins_enc[i] = [np.zeros(len(join2vec)+2)]
                join_selectivity_enc[i] = [np.zeros(2+2*self.histogram_dim)]

        return join_selectivity_enc, joins_enc, predicates_enc

    def get_train_test_dataset(self, num_queries=10000, file_name_data=None):
        file_name_data = self.file_name_train_data if file_name_data is None else file_name_data
        joins, predicates, labels = self.load_data(file_name_data, self.num_materialized_samples, num_queries=num_queries)
        # Get column name dict
        self.column2vec, self.idx2column = get_set_encoding(self.columns)

        # Get join dict
        self.join2vec, self.idx2join = get_set_encoding(self.joins)

        # Get op dict
        ops = ["<", ">", "=", "<>"]
        self.op2vec, self.idx2op = get_set_encoding(ops)

        # init scalar
        self.load_table_size_scalar()
        self.load_n_unique_scalar()

        # Encode dataset with metas
        enc_join_selectivities, enc_join_metas, enc_preds = self.encode_dataset(joins, predicates)
        norm_labels, min_label_val, max_label_val = normalize_labels(labels)

        # Split in training and validation samples
        num_train = int(num_queries * 0.9)
        num_test = num_queries - num_train

        # samples_train = samples_enc[:num_train]
        predicates_train = enc_preds[:num_train]
        joins_selectivity_train = enc_join_selectivities[:num_train]
        joins_train = enc_join_metas[:num_train]
        labels_train = norm_labels[:num_train]

        # samples_test = samples_enc[num_train:num_train + num_test]
        predicates_test = enc_preds[num_train:num_train + num_test]
        joins_selectivity_test = enc_join_selectivities[num_train:num_train + num_test]
        joins_test = enc_join_metas[num_train:num_train + num_test]
        labels_test = norm_labels[num_train:num_train + num_test]

        print("Number of training samples: {}".format(len(labels_train)))
        print("Number of validation samples: {}".format(len(labels_test)))

        max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
        max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

        train_data = [predicates_train, joins_selectivity_train, joins_train]
        test_data = [predicates_test, joins_selectivity_test, joins_test]
        return labels_train, labels_test, min_label_val, max_label_val, max_num_joins, max_num_predicates, train_data, test_data

    def get_and_encode_outside_dataset(self, file_name_data):
        joins, predicates, labels = self.load_data(file_name_data, self.num_materialized_samples)
        enc_join_selectivities, enc_join_metas, enc_preds = self.encode_dataset(joins, predicates)
        enc_dataset = [enc_preds, enc_join_selectivities, enc_join_metas]
        return labels, enc_dataset

    def make_dataset(self, predicates, joins_selectivity, joins, labels, max_num_joins, max_num_predicates):
        predicate_masks = []
        predicate_tensors = []
        for idx, predicate in enumerate(predicates):
            predicate_tensor = np.vstack(predicate)
            num_pad = max_num_predicates - predicate_tensor.shape[0]
            predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
            predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
            predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
            predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
            predicate_masks.append(np.expand_dims(predicate_mask, 0))
        predicate_tensors = np.vstack(predicate_tensors)
        predicate_tensors = torch.FloatTensor(predicate_tensors)
        predicate_masks = np.vstack(predicate_masks)
        predicate_masks = torch.FloatTensor(predicate_masks)

        join_masks = []
        join_tensors = []
        selectivity_tensors = []
        for idx, join in enumerate(joins):
            join_tensor = np.vstack(join)
            selectivity_tensor = np.vstack(joins_selectivity[idx])
            num_pad = max_num_joins - join_tensor.shape[0]
            join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
            join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
            selectivity_tensor = np.pad(selectivity_tensor, ((0, num_pad), (0, 0)), 'constant')
            join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
            join_tensors.append(np.expand_dims(join_tensor, 0))
            selectivity_tensors.append(np.expand_dims(selectivity_tensor, 0))
            join_masks.append(np.expand_dims(join_mask, 0))
        join_tensors = np.vstack(join_tensors)
        join_tensors = torch.FloatTensor(join_tensors)
        selectivity_tensors = np.vstack(selectivity_tensors)
        selectivity_tensors = torch.FloatTensor(selectivity_tensors)
        join_masks = np.vstack(join_masks)
        join_masks = torch.FloatTensor(join_masks)

        target_tensor = torch.FloatTensor(labels)

        return dataset.TensorDataset(predicate_tensors, selectivity_tensors, join_tensors, target_tensor,
                                     predicate_masks, join_masks)


def get_torch_train_data():
    data_loader = DataLoader("./GenData/data_gen.csv", "./GenData/gen_data_meta.pkl", 100)
    labels_train, labels_test, min_label_val, max_label_val,  max_num_joins, max_num_predicates, train_data, test_data = \
        data_loader.get_train_test_dataset(num_queries=10000)
    input_dim = [data_loader.pred_dim, data_loader.selectivity_dim, data_loader.join_dim]
    train_dataset = data_loader.make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                             max_num_predicates=max_num_predicates)
    test_dataset = data_loader.make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
                                            max_num_predicates=max_num_predicates)
    return train_dataset, test_dataset, input_dim, data_loader, min_label_val, max_label_val, labels_train, labels_test


if __name__ == "__main__":
    pass
