fp = open("./data_gen.sql", "w+")
for line in open("./data_gen.csv"):
    parts = line.strip().split("#")
    sql = "SELECT count(*) FROM "
    tables = parts[0]
    sql += tables
    sql += " WHERE "
    joins = parts[1].strip().split(",") if parts[1] != "" else []
    predicates_parts = parts[2].strip().split(",") if parts[2] != "" else []
    predicates = []
    for i in range(0, len(predicates_parts), 3):
        pred = predicates_parts[i] + predicates_parts[i+1] + predicates_parts[i+2]
        predicates.append(pred)
    jp = " AND ".join(joins + predicates)
    sql += jp
    print(sql)
    fp.write(sql + "\n")
fp.close()
