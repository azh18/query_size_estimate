import csv
import numpy as np
from traditional.util import chunks, encode_data_fixed, normalize_labels
import pickle
from traditional.util import get_set_encoding, get_all_joins,get_all_column_names
import argparse

def load_column_names(file_name):
    column_names = pickle.load(open(file_name, "rb"))
    return column_names


def load_data(file_name, num_materialized_samples):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    # Load queries
    with open(file_name + ".csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")

    # # Load bitmaps
    # num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    # with open(file_name + ".bitmaps", 'rb') as f:
    #     for i in range(len(tables)):
    #         four_bytes = f.read(4)
    #         if not four_bytes:
    #             print("Error while reading 'four_bytes'")
    #             exit(1)
    #         num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
    #         bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
    #         for j in range(num_bitmaps_curr_query):
    #             # Read bitmap
    #             bitmap_bytes = f.read(num_bytes_per_bitmap)
    #             if not bitmap_bytes:
    #                 print("Error while reading 'bitmap_bytes'")
    #                 exit(1)
    #             bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
    #         samples.append(bitmaps)
    # print("Loaded bitmaps")

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]

    return joins, predicates, tables, samples, label


def load_and_encode_train_data(num_materialized_samples):
    # file_name_queries = "data/train"
    file_name_queries = "GenData/data_gen"

    file_name_column_min_max_vals = "GenData/data_meta.csv"

    joins, predicates, tables, samples, label = load_data(file_name_queries, num_materialized_samples)

    # Get column name dict
    # column_names = load_column_names("../db_metas/columns.pkl")
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    # table_names = get_all_table_names(tables)
    # table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    # operators = get_all_operators(predicates)
    # operators = ["<", ">", "="]
    # op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(file_name_column_min_max_vals, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}  # col_name -> [min, max]
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    # samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data_fixed(predicates, joins, column_min_max_vals, column_names, join2vec,
                                                  len(column_names), len(join_set))
    # predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)

    return predicates_enc, joins_enc, label

