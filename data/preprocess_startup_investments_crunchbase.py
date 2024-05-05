import csv
import os
import re

from replace_map import REPLACE_MAP

startup_investments_crunchbase_file_path = os.path.join(os.path.dirname(__file__), os.path.join("raw", "startup_investments_crunchbase.csv"))

names = []
with open(startup_investments_crunchbase_file_path, "r", errors="ignore") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        name = row[1].strip()
        names.append(name)

out_rows = []
for n in names:
    for s1, s2 in REPLACE_MAP.items():
        n = n.replace(s1, s2).strip()
    n = n.replace(".", "").strip()
    n = re.sub("^&", "", n).strip()
    n = re.sub("\\([^)]*\\)", "", n).strip()
    n = re.sub("# ?\\d+", "", n).strip()
    n = re.sub("\\d\\d\\d\\d+", "", n).strip()
    n = re.sub(" +", " ", n).strip()
    if n:
      out_rows.append([n])

out_file_path = os.path.join(os.path.dirname(__file__), "startup_investments_crunchbase.csv")

with open(out_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out_rows)
