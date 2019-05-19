import csv
import torch
from torch.utils.data import dataset
import pickle

from mscnNoSample.util import *


def make_multiple_slice(n_slice):
    ret = []
    for i in range(n_slice):
        ret.append([])
    return ret


class Scalar:
    def __init__(self, min_val=0, max_val=0):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data_list):
        self.min_val = min(data_list)
        self.max_val = max(data_list)

    def transform(self, data_list):
        ret = []
        for d in data_list:
            ret.append((d-self.min_val)/(self.max_val-self.min_val))
        return ret

    def inverse_transform(self, data_list):
        ret = []
        for d in data_list:
            val = d * (self.max_val-self.min_val) + self.min_val
            ret.append(val)
        return ret


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
        self.joins = make_multiple_slice(5)  # (raw_joins_list, joins_onehot_list, table_size_list, hist1_list, hist2_list, col_meta_list)
        self.predicates = make_multiple_slice(3)  # (raw_preds_list, preds_onehot_list, estimated_hit_rate, table_size)
        self.labels = make_multiple_slice(1)
        self.file_name_train_data = file_name_train_data
        self.num_materialized_samples = num_materialized_samples
        # define scalars
        self.join_hist_scalar = Scalar()
        self.table_size_scalar = Scalar()
        self.col_val_scalar_dict = {}

    def load_data(self, file_name, num_materialized_samples):
        joins = []
        predicates = []
        labels = []
        # Load queries
        with open(file_name, 'rU') as f:
            data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
            for row in data_raw:
                joins.append(row[1].split(','))
                predicates.append(row[2].split(','))
                if int(row[3]) < 1:
                    print("Queries must have non-zero cardinalities")
                    exit(1)
                labels.append(row[3])
        print("Loaded queries")

        # Split predicates
        predicates = [list(chunks(d, 3)) for d in predicates]

        return joins, predicates, labels

    def load_table_size_scalar(self):
        min_val = min(self.table_abbr_size.values())
        max_val = max(self.table_abbr_size.values())
        self.table_size_scalar = Scalar(min_val=min_val, max_val=max_val)
        return self.table_size_scalar

    def load_col_value_scalar(self, col):
        if col not in self.col_val_scalar_dict:
            self.col_val_scalar_dict[col] = Scalar(min_val=self.col_min_max[col][0], max_val=self.col_min_max[col][1])
        return self.col_val_scalar_dict[col]

    def encode_dataset(self, joins, predicates, column2vec, idx2column, join2vec, idx2join, op2vec, idx2op):
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
                    val = predicate[2]
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
                    if not lower_bound or not upper_bound:
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
                norm_ticks = self.col_val_scalar_dict[col].transform(self.raw_histogram[0])
                norm_hists = self.raw_histogram[1] / max(self.raw_histogram[1])
                estimate_hit_rate = 0
                if op == "<>":
                    lower_bound = pred_block[1]
                    scale = pred_block[2] - lower_bound
                    beg_idx, end_idx = -1, -1
                    for idx in range(len(self.raw_histogram[1])):
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
                    for idx in range(len(self.raw_histogram[1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        if lb <= lower_bound < ub:
                            estimate_hit_rate += norm_hists[idx]
                            break
                elif op == "<":
                    lower_bound = 0
                    scale = pred_block[1]
                    for idx in range(len(self.raw_histogram[1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        estimate_hit_rate += norm_hists[idx]
                        if lb <= lower_bound < ub:
                            break
                else:  # op == >
                    lower_bound = pred_block[1]
                    scale = 1 - lower_bound
                    beg_idx = -1
                    for idx in range(len(self.raw_histogram[1])):
                        lb = norm_ticks[idx]
                        ub = norm_ticks[idx + 1]
                        if lb <= lower_bound < ub:
                            beg_idx = idx
                    estimate_hit_rate += sum(norm_hists[beg_idx:])

                pred_vec = []
                pred_vec.append(op2vec[op])
                pred_vec.append(column2vec[col])
                pred_vec.append([lower_bound, scale, estimate_hit_rate, norm_table_size])
                pred_vec = np.hstack(pred_vec)
                predicates_enc[i].append(pred_vec)

            # Join instruction
            for join in joins[i]:
                col1, col2 = join.split("=")
                tab1, tab2 = col1.split(".")[0], col2.split(".")[0]
                col1_n_unique, col2_n_unique, col1_min, col2_min, col1_max, col2_max = \
                    self.col_n_unique_values[col1], self.col_n_unique_values[col2],\
                    self.col_min_max[col1][0], self.col_min_max[col2][0], \
                    self.col_min_max[col1][1], self.col_min_max[col2][1]
                col1_ticks, col1,hist = self.raw_histogram[col1]
                col2_ticks, col2,hist = self.raw_histogram[col2]
                col_min, col_max = min(col1_min, col2_min), max(col1_max, col2_max)
                # regenerate histogram
                # TODO: how to merge two histogram with different ticks?

                # TODO: construct join_vec


                join_vec = join2vec[predicate]
                joins_enc[i].append(join_vec)
        return predicates_enc, joins_enc

    def get_train_test_dataset(self, file_name_data=None):
        file_name_data = self.file_name_train_data if file_name_data is None else file_name_data
        joins, predicates, labels = self.load_data(file_name_data, self.num_materialized_samples)
        # Get column name dict
        column2vec, idx2column = get_set_encoding(self.columns)

        # Get join dict
        join2vec, idx2join = get_set_encoding(self.joins)

        # Get op dict
        ops = ["<", ">", "=", "<>"]
        op2vec, idx2op = get_set_encoding(ops)

        # init scalar
        self.load_table_size_scalar()

        # Encode dataset with metas
        enc_join_selectivities, enc_join_metas, enc_preds = self.encode_dataset(joins, predicates, column2vec, idx2column, join2vec, idx2join, op2vec, idx2op)

def load_and_encode_train_data(num_queries, num_materialized_samples):
    file_name_queries = "GenData/data_gen.csv"
    file_name_metas = "GenData/gen_data_meta.pkl"

    joins, predicates, tables, samples, label = load_data(file_name_queries, num_materialized_samples)

    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(file_name_column_min_max_vals, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    # samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    # samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    # samples_test = samples_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [predicates_train, joins_train]
    test_data = [predicates_test, joins_test]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data


def make_dataset(predicates, joins, labels, max_num_joins, max_num_predicates):
    """Add zero-padding and wrap as tensor dataset."""

    # sample_masks = []
    # sample_tensors = []
    # for sample in samples:
    #     sample_tensor = np.vstack(sample)
    #     num_pad = max_num_joins + 1 - sample_tensor.shape[0]
    #     sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
    #     sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
    #     sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
    #     sample_tensors.append(np.expand_dims(sample_tensor, 0))
    #     sample_masks.append(np.expand_dims(sample_mask, 0))
    # sample_tensors = np.vstack(sample_tensors)
    # sample_tensors = torch.FloatTensor(sample_tensors)
    # sample_masks = np.vstack(sample_masks)
    # sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
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
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(predicate_tensors, join_tensors, target_tensor,
                                 predicate_masks, join_masks)


def get_train_datasets(num_queries, num_materialized_samples):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, test_data = load_and_encode_train_data(
        num_queries, num_materialized_samples)
    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_dataset, test_dataset
