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
        conn = pymysql.connect(host='35.200.254.32', user=username, passwd=password, db='imdb', read_timeout=600)
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

        # self.abbr2table.pop("mi_idx")
        # self.table2abbr.pop("movie_info_idx")
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
        cand_ops = ["=", ">", "<", "<>"]
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
                if len(chosen_idx) == len(cand_cols):
                    # if no more pred can be generated, break
                    break
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
                op = cand_ops[random.randint(0, 3)]

                vals = []

                rand_row_sql = "SELECT x.%s from %s as x JOIN (SELECT FLOOR(RAND()*(SELECT MAX(id) FROM %s)) as id) as y WHERE x.id = y.id" % (col_single_name, tab, tab)
                # rand_row_sql = "SELECT %s FROM %s WHERE id = FLOOR(RAND()*(select max(id) from %s)) AND %s is not null LIMIT 1" % (col, tab, tab, col)
                try:
                    sql_cur.execute(rand_row_sql)
                    val = sql_cur.fetchone()
                except Exception as e:
                    print(e)
                    conn.ping(reconnect=True)
                    continue
                if type(val) is tuple:
                    val = val[0]
                if val is None:
                    print("val is None!!!")
                    print(rand_row_sql)
                    continue
                vals.append(val)
                # if <>, generate second value
                if op == "<>":
                    val2 = None
                    while val2 is None or val2 == val:
                        rand_row_sql = "SELECT x.%s from %s as x JOIN (SELECT FLOOR(RAND()*(SELECT MAX(id) FROM %s)) as id) as y WHERE x.id = y.id" % (
                        col_single_name, tab, tab)
                        # rand_row_sql = "SELECT %s FROM %s WHERE id = FLOOR(RAND()*(select max(id) from %s)) AND %s is not null LIMIT 1" % (col, tab, tab, col)
                        sql_cur.execute(rand_row_sql)
                        val2 = sql_cur.fetchone()
                        if type(val2) is tuple:
                            val2 = val2[0]
                        if val2 is None:
                            print("val is None!!!")
                            print(rand_row_sql)
                            continue
                    vals.append(val2)
                if op == "<>":
                    vals = sorted(vals)
                    pred1 = "%s%s%s" % (col, ">", vals[0])
                    pred2 = "%s%s%s" % (col, "<", vals[1])
                    predicates += [pred1, pred2]
                    pred_items += [(col, ">", str(vals[0])), (col, "<", str(vals[1]))]
                else:
                    val = vals[0]
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


class MysqlConnect:
    conn = None

    @staticmethod
    def get_conn():
        username = 'zbw0046'
        password = '123456'
        if MysqlConnect.conn is None:
            MysqlConnect.conn = pymysql.connect(host='35.200.254.32', user=username, passwd=password, db='imdb', read_timeout=60)
        else:
            MysqlConnect.conn.ping(reconnect=True)
        return MysqlConnect.conn


class MetadataFetcher(SqlElemGenerator):
    def __init__(self, train_data_sql):
        super(MetadataFetcher, self).__init__()
        self.table_size = {}  # tab->size
        self.histograms = {}  # col->[hist1, hist2, ...]
        self.n_unique_values = {}  # col->n_unique_values
        self.feed(train_data_sql)
        self.connManager = MysqlConnect

    def get_table_size(self):
        for tbl in self.table2abbr.keys():
            table_abbr = self.table2abbr[tbl]
            conn = self.connManager.get_conn()
            cursor = conn.cursor()
            sql = "SELECT count(*) from %s" % tbl
            print(sql)
            for retry in range(3):
                try:
                    cursor.execute(sql)
                    ret = cursor.fetchone()
                    if type(ret) is tuple:
                        ret = ret[0]
                    if ret is None:
                        continue
                    self.table_size[table_abbr] = ret
                    break
                except Exception as e:
                    print(e)
        return self.table_size

    def get_n_unique_vals(self):
        for col in self.columns:
            tbl_abbr = col.split(".")[0]
            tbl = self.abbr2table[tbl_abbr]
            single_col = col.split(".")[1]
            sql = "SELECT count(distinct(%s)) FROM %s" % (
                single_col, tbl
            )
            for retry in range(3):
                conn = self.connManager.get_conn()
                cursor = conn.cursor()
                print(sql)
                try:
                    cursor.execute(sql)
                    ret = cursor.fetchone()
                    if type(ret) is tuple:
                        ret = ret[0]
                    if ret is None:
                        continue
                    self.n_unique_values[col] = ret
                    break
                except Exception as e:
                    print(e)
        return self.n_unique_values

    def get_histograms(self):
        for col in self.columns:
            hist = self.get_col_histogram(col)
            self.histograms[col] = hist
        return self.histograms

    def get_col_histogram(self, tbl_col):
        N_INTERVAL = 10
        min_val, max_val = self.columns_dom[tbl_col]
        levels = []
        interval = (max_val - min_val) / N_INTERVAL
        for i in range(N_INTERVAL):
            levels.append(int(min_val + i * interval))
        levels.append(max_val)
        hist = []
        tbl_abbr = tbl_col.split(".")[0]
        tbl = self.abbr2table[tbl_abbr]
        col = tbl_col.split(".")[1]
        for i in range(N_INTERVAL):
            min_bar, max_bar = levels[i], levels[i+1]
            sql = "SELECT count(*) FROM %s WHERE %s >= %d AND %s <= %d" % (
                tbl, col, min_bar, col, max_bar
            )
            for retry in range(3):
                conn = self.connManager.get_conn()
                cursor = conn.cursor()
                print(sql)
                try:
                    cursor.execute(sql)
                    ret = cursor.fetchone()
                    if type(ret) is tuple:
                        ret = ret[0]
                    if ret is None:
                        continue
                    hist.append(ret)
                    break
                except Exception as e:
                    print(e)
        print(hist)
        return hist

if __name__ == "__main__":
    fetcher = MetadataFetcher("./data_gen.sql")
    table_size = fetcher.get_n_unique_vals()
    print(table_size)
