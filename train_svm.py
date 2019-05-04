import argparse
from traditional.data import load_and_encode_train_data
from traditional.svm_model import SVRModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", help="synthetic, scale, or job-light", type=str, default="synthetic")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    args = parser.parse_args()
    train_data, labels_train, test_data, labels_test, column_min_max_vals, label_min_max_vals = \
        load_and_encode_train_data(args.queries, 1000)
    SVR_Model = SVRModel()
    SVR_Model.bind_dataset(train_data, labels_train, test_data, labels_test, label_min_max_vals)
    SVR_Model.train_grid(["scale", "auto"], [1,0.9, 0.8,0.7, 0.6, 0.5, 0.4,0.3, 0.2, 0.1])
