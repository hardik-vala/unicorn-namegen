import csv
import os
import re

from replace_map import REPLACE_MAP

odbus_file_path = os.path.join(os.path.dirname(__file__), os.path.join("raw", os.path.join("ODBus_v1", "ODBus_v1.csv")))

rows = []
with open(odbus_file_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 6:
            continue
        name = row[1].strip()
        description = row[5].strip()
        rows.append([name, description])

for r in rows:
    for s1, s2 in REPLACE_MAP.items():
        r[0] = r[0].replace(s1, s2).strip()
    r[0] = r[0].replace(".", "").strip()
    r[0] = re.sub("\\([^)]+\\)", "", r[0]).strip()
    r[0] = re.sub("# ?\\d+", "", r[0]).strip()
    r[0] = re.sub("\\d\\d\\d+", "", r[0]).strip()
    r[0] = re.sub(" +", " ", r[0]).strip()
    if len(r[0]) > 3:
      r[0] = r[0].title()

out_file_path = os.path.join(os.path.dirname(__file__), "odbus_v1.csv")

with open(out_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
