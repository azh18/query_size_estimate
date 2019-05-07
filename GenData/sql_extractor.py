# extract tables, columns, joins from sql
import pymysql
import pickle
import random
import csv


class SqlElemGenerator:
    def __init__(self):
        self.table2abbr = {}
        self.abbr2table = {}
        self.columns = set()
        self.joins = set()
        self.columns_dom = {}  # col->(min, max)
        self.sql_conn = None

    # add some known cols
    def self_augment(self):
        extra_cols = ["mc.movie_id", "ci.movie_id", "mi.movie_id", "mk.movie_id"]
        for c in extra_cols:
            self.columns.add(c)

    def connect_db(self):
        username = 'zbw0046'
        password = '123456'
        conn = pymysql.connect(host='35.200.254.32', user=username, passwd=password, db='imdb', read_timeout=60)
        self.sql_conn = conn
        return conn

    def dump(self, pk_file):
        pickle.dump((self.table2abbr, self.abbr2table, self.columns, self.joins, self.columns_dom), open(pk_file, "wb"))

    def load(self, pk_file):
        self.table2abbr, self.abbr2table, self.columns, self.joins, self.columns_dom = pickle.load(open(pk_file, "rb"))

    def extract_elements_from_sql(self, sql):
        table2abbr = {}
        abbr2table = {}
        joins = set()
        columns = set()
        # extract tables
        sql = sql.strip(";")
        table_sql = sql.split("FROM")[1].split("WHERE")[0]
        table_elems = table_sql.strip().split(",")
        for t in table_elems:
            table_name, table_abbr = t.strip().split(" ")
            table2abbr[table_name] = table_abbr
            abbr2table[table_abbr] = table_name
        # extract joins and columns
        where_sql = sql.split("WHERE")[1]
        where_elems = where_sql.strip().split("AND")
        for w in where_elems:
            w = w.strip()
            if "=" in w:
                left_elem, right_elem = w.split("=")
                if "." in right_elem and right_elem.split(".")[0] in abbr2table:
                    # indicate that this is a join
                    joins.add(w.strip())
                    continue
            # it is a predicate if it is not a join
            op = ">"
            for cand in [">", "<", "="]:
                if cand in w:
                    op = cand
                    break
            left_elem, right_elem = w.split(op)
            col = left_elem.strip()
            columns.add(col)
        # finish reading sql, merge
        self.table2abbr.update(table2abbr)
        self.abbr2table.update(abbr2table)
        self.columns = self.columns.union(columns)
        self.joins = self.joins.union(joins)

    def feed(self, sql_file):
        for line in open(sql_file):
            self.extract_elements_from_sql(line)
        # mi_idx is not in our db, so delete it
        invalid = []
        for c in self.joins:
            if "mi_idx" in c:
                invalid.append(c)
        for i in invalid:
            self.joins.remove(i)
        invalid = []
        for c in self.columns:
            if "mi_idx" in c:
                invalid.append(c)
        for i in invalid:
            self.columns.remove(i)

        self.abbr2table.pop("mi_idx")
        self.table2abbr.pop("movie_info_idx")
        self.self_augment()
        self.get_col_val_min_max()


    def show(self):
        print(self.table2abbr)
        print(self.columns)
        print(self.joins)
        print(self.columns_dom)

    def get_col_val_min_max(self):
        conn = self.sql_conn if self.sql_conn is not None else self.connect_db()
        cur = conn.cursor()
        for col in self.columns:
            table_abbr = col.split(".")[0]
            table_name = self.abbr2table[table_abbr]
            sql = "SELECT min(%s), max(%s) from %s %s" % (col, col, table_name, table_abbr)
            print(sql)
            cur.execute(sql)
            min_val, max_val = cur.fetchone()
            self.columns_dom[col] = (min_val, max_val)
            print(self.columns_dom)

    def generate_sqls(self, num, max_n_join, max_n_pred, min_n_pred=1):
        queries_output = []
        cand_joins = list(self.joins)
        table_column_map = {}
        for c in self.columns:
            tbl_abbr = c.split(".")[0]
            if tbl_abbr in table_column_map:
                table_column_map[tbl_abbr].append(c)
            else:
                table_column_map[tbl_abbr] = [c]
        cand_tables = list(table_column_map.keys())
        cand_ops = ["=", ">", "<"]
        conn = self.sql_conn if self.sql_conn is not None else self.connect_db()
        sql_cur = conn.cursor()
        for i in range(num):
            table_associated_abbr = set()
            n_join = random.randint(0, max_n_join)
            n_pred = random.randint(min_n_pred, max_n_pred)
            # random choose joins
            chosen_idx = set()
            while len(chosen_idx) < n_join:
                idx = random.randint(0, len(self.joins)-1)
                chosen_idx.add(idx)
            chosen_joins = list(map(lambda x: cand_joins[x], chosen_idx))
            for j in chosen_joins:
                for e in j.split("="):
                    table_associated_abbr.add(e.split(".")[0])
            # random choose cols from associated tables

            if n_join == 0:
                # if no joins, randomly select cols from a table
                # randomly choose a table
                cand_table = cand_tables[random.randint(0, len(cand_tables)-1)]
                cand_cols = table_column_map[cand_table]
            else:
                # select cols from join associated tables
                cand_cols = []
                for tbl_abbr in table_associated_abbr:
                    cand_cols += table_column_map[tbl_abbr]

            chosen_idx = set()
            while len(chosen_idx) < n_pred:
                idx = random.randint(0, len(cand_cols)-1)
                chosen_idx.add(idx)
            chosen_cols = list(map(lambda x: cand_cols[x], chosen_idx))
            for c in chosen_cols:
                table_associated_abbr.add(c.split(".")[0])
            # combine predicates
            predicates = []
            pred_items = []
            for col in chosen_cols:
                # random choose row to form value
                tab_abbr = col.split(".")[0].strip()
                tab = self.abbr2table[tab_abbr]
                col_single_name = col.split(".")[1]
                rand_row_sql = "SELECT x.%s from %s as x JOIN (SELECT FLOOR(RAND()*(SELECT MAX(id) FROM %s)) as id) as y WHERE x.id = y.id" % (col_single_name, tab, tab)
                # rand_row_sql = "SELECT %s FROM %s WHERE id = FLOOR(RAND()*(select max(id) from %s)) AND %s is not null LIMIT 1" % (col, tab, tab, col)
                sql_cur.execute(rand_row_sql)
                val = sql_cur.fetchone()
                if type(val) is tuple:
                    val = val[0]
                if val is None:
                    print("val is None!!!")
                    print(rand_row_sql)
                    continue
                op = cand_ops[random.randint(0, 2)]
                pred = "%s%s%s" % (col, op, val)
                predicates.append(pred)
                pred_items.append((col, op, str(val)))
            if len(chosen_joins) + len(predicates) == 0:
                continue
            where_block = " AND ".join(chosen_joins + predicates)
            tables = list(map(lambda x: "%s %s" % (self.abbr2table[x], x), table_associated_abbr))
            table_block = ",".join(tables)
            gen_sql = "SELECT count(*) FROM %s WHERE %s;" % (table_block, where_block)
            print(gen_sql)
            query_item = (tables, chosen_joins, pred_items, gen_sql)
            queries_output.append(query_item)
        return queries_output


class DatasetGenerator:
    sql_conn = None

    def __init__(self, csv_file):
        DatasetGenerator.init_sql_conn()
        self.fp = open(csv_file, "a+")

    @staticmethod
    def init_sql_conn():
        if DatasetGenerator.sql_conn is None:
            username = 'zbw0046'
            password = '123456'
            conn = pymysql.connect(host='35.200.254.32', user=username, passwd=password, db='imdb', read_timeout=60)
            DatasetGenerator.sql_conn = conn
            return conn

    # get the csv format of a query and its result
    @staticmethod
    def get_csv_query(query_item, result_size):
        tables, joins, preds, sql = query_item
        pred_items = []
        for pred in preds:
            pred_items.append(",".join(pred))
        csv_str = "%s#%s#%s#%d" % (",".join(tables), ",".join(joins), ",".join(pred_items), result_size)
        return csv_str

    # use mysql to run a query and get query size result
    def feed_sql(self, sql):
        conn = DatasetGenerator.sql_conn
        sql_cur = conn.cursor()
        try:
            sql_cur.execute(sql)
            result = sql_cur.fetchone()
            if len(result) != 1:
                conn.ping(reconnect=True)
                return None
            print(result)
        except Exception as e:
            print(e)
            conn.ping(reconnect=True)
            return None
        result = result[0]
        return result

    # handler function, to generate a dataset for given query_items
    def generate_dataset(self, query_items):
        for item in query_items:
            sql = item[3]
            res = self.feed_sql(sql)
            if res is None or res == 0:
                continue
            csv_item = DatasetGenerator.get_csv_query(item, res)
            self.fp.write(csv_item + "\n")
            self.fp.flush()
            print(csv_item)


if __name__ == "__main__":
    sql_dir = "../workloads/"
    sql_files = [
        "job-light.sql",
        "synthetic.sql",
        "scale.sql"
    ]
    extractor = SqlElemGenerator()
    # extractor.load("sql_info.pkl")
    for sf in sql_files:
        extractor.feed(sql_dir + sf)
    extractor.dump("sql_info.pkl")
    extractor.show()
    query_items = extractor.generate_sqls(1000, 1, 2)
    data_generator = DatasetGenerator("data_gen.csv")
    data_generator.generate_dataset(query_items)
