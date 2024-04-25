import csv
import os
import re

replace_map = {
    "®": "",
    "á": "a",
    "ã": "a",
    "é": "e",
    "Ē": "E",
    "–": "-",
    "𐂂": "",
    "💰": "",
    "💸": "",
    "🦜": "",
    "%": "",
    "!": "",
    "'": "",
    ",": "",
    "Inc.": "",
    "inc.": "",
    "Inc": "",
    "inc": "",
    " /": " ",
    "/": "",
}

yc_companies_file_path = os.path.join(os.path.dirname(__file__), os.path.join("raw", "yc_companies.csv"))

rows = []
with open(yc_companies_file_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    for row in reader:
        name = row[1].strip()
        description = row[3].strip()
        rows.append([name, description])

for r in rows:
    for s1, s2 in replace_map.items():
        r[0] = r[0].replace(s1, s2).strip()
    r[0] = re.sub("\\([^)]+\\)", "", r[0]).strip()

out_file_path = os.path.join(os.path.dirname(__file__), "yc_companies.csv")

with open(out_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
