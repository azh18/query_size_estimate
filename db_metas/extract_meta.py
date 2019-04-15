import sys
import pickle

FILE_NAME = "imdb_2019-04-14.sql"

sql_fp = open(FILE_NAME, "r")
table_name = ""
columns = {}

while True:
    line = sql_fp.readline()
    if not line:
        break
    line = line.rstrip()
    if "CREATE TABLE" in line:
        metas = line.split("`")
        table_name = metas[1]
        continue
    metas = line.split("`")
    if len(metas) < 2:
        continue
    if metas[0] == "  ":
        column_name = metas[1]
        if table_name not in columns:
            columns[table_name] = [table_name + "." + column_name]
        else:
            columns[table_name].append(table_name + "." + column_name)

all_columns = []
for k in columns:
    all_columns += columns[k]
pickle.dump(all_columns, open("columns.pkl", "wb"))





