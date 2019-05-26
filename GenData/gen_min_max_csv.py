# name, min, max, cardinality, num_unique_values
import pickle
pkl_meta_file_name = "gen_data_meta.pkl"
data_meta = pickle.load(open(pkl_meta_file_name, "rb"))
output_file_fp = open("data_meta.csv", "w")
output_file_fp.write(",".join(["name", "min", "max", "cardinality", "num_unique_values"]))
output_file_fp.write("\n")
columns = data_meta[2]
min_max_vals = data_meta[4]
cardinalities = data_meta[5]
unique_values_dict = data_meta[7]
for col in columns:
    name = col
    min_val, max_val = min_max_vals[col]
    tab_abbr = col.split(".")[0]
    cardinality = cardinalities[tab_abbr]
    num_unique_values = unique_values_dict[col]
    output_file_fp.write(",".join([name, str(min_val), str(max_val), str(cardinality), str(num_unique_values)]))
    output_file_fp.write("\n")
    output_file_fp.flush()
output_file_fp.close()
