import numpy as np

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot

def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing

def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set

def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def encode_data_fixed(predicates, joins, column_min_max_vals, columns, join2vec, n_total_cols, n_total_joins):
    predicates_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        attr_preds = {}
        for predicate in query:
            print(query)
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                if column not in attr_preds:
                    attr_preds[column] = np.array([0, 0, 1], dtype=np.float64)

                attr_preds[column][0] = 1
                if operator == "<":
                    if norm_val < attr_preds[column][2]:
                        attr_preds[column][2] = norm_val
                elif operator == ">":
                    if norm_val > attr_preds[column][1]:
                        attr_preds[column][1] = norm_val
                elif operator == "=":
                    attr_preds[column][1] = norm_val
                    attr_preds[column][2] = norm_val
                else:
                    attr_preds[column][0] = 0
        this_predicates_enc = np.array([])
        for col in columns:
            if col not in attr_preds:
                this_predicates_enc = np.hstack([this_predicates_enc, np.array([0, 0, 0], dtype=np.float64)])
            else:
                diff_range = attr_preds[col][2] - attr_preds[col][1]
                attr_preds[col][2] = diff_range
                if attr_preds[col][0] == 0:
                    attr_preds[col] = np.array([0, 0, 0], dtype=np.float64)
                this_predicates_enc = np.hstack([this_predicates_enc, attr_preds[col]])
        print(this_predicates_enc)
        predicates_enc.append(this_predicates_enc)

        this_joins_enc = np.zeros(n_total_joins)
        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            this_joins_enc += join_vec

        joins_enc.append(this_joins_enc)

    return predicates_enc, joins_enc

def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)

def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)
