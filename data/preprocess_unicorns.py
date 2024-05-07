import csv
import os
import re

from replace_map import REPLACE_MAP

unicorns_file_path = os.path.join(
    os.path.dirname(__file__), os.path.join("raw", "unicorns_till_sep_2022.csv")
)

names = []
with open(unicorns_file_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        name = row[0].strip()
        names.append(name)

out_rows = []
for n in names:
    for s1, s2 in REPLACE_MAP.items():
        n = n.replace(s1, s2).strip()
    n = re.sub("\\([^)]+\\)", "", n).strip()
    if n:
        out_rows.append([n])

out_file_path = os.path.join(os.path.dirname(__file__), "unicorns.csv")

with open(out_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out_rows)
