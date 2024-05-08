import csv
import os
import re

gd_auctions_export_file_path = os.path.join(os.path.dirname(__file__), os.path.join("raw", "gd_auctions_export_1715053158020.csv"))

domain_names = []
with open(gd_auctions_export_file_path, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        dname = row[1].strip()
        domain_names.append(dname)

out_rows = []
for dn in domain_names:
    name = dn.split(".")[0]
    
    if len(name) < 5 or any(c.isdigit() for c in name):
        continue
    
    out_rows.append([name.title()])

out_file_path = os.path.join(os.path.dirname(__file__), "gd_auctions_export.csv")

with open(out_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out_rows)
