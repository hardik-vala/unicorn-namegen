import csv
import numpy as np
import os
import pickle
import random


# ------------------------------------------------------------------------------
with_odbus_v1 = True
max_odbus_v1_names = 12000
with_sec_edgar_names = True
max_sec_edgar_names = 12000
split_frac = 0.9
# ------------------------------------------------------------------------------

# read the dataset
def read_names(file_path):
    with open(file_path, "r") as f:
        return [l.strip() for l in f.readlines()]

def read_csv_names(file_path):
    names = []
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            name = row[0]
            names.append(name)
    return names

names_file_path = os.path.join(os.path.dirname(__file__), "names.txt")
data = read_names(names_file_path)

yc_companies_file_path = os.path.join(os.path.dirname(__file__), "yc_companies.csv")
yc_names = read_csv_names(yc_companies_file_path)
data.extend(yc_names)

startup_investments_crunchbase = os.path.join(os.path.dirname(__file__), "startup_investments_crunchbase.csv")
cb_startup_names = read_names(startup_investments_crunchbase)
data.extend(cb_startup_names)

if with_odbus_v1:
    odbus_v1_file_path = os.path.join(os.path.dirname(__file__), "odbus_v1.csv")
    odbus_v1_names = read_csv_names(odbus_v1_file_path)
    if max_odbus_v1_names:
        odbus_v1_names = random.sample(odbus_v1_names, k=max_odbus_v1_names)
    data.extend(odbus_v1_names)

if with_sec_edgar_names:
    sec_edgar_names_file_path = os.path.join(os.path.dirname(__file__), "sec__edgar_company_names.txt")
    sec_edgar_names = read_names(sec_edgar_names_file_path)
    if max_sec_edgar_names:
        sec_edgar_names = random.sample(sec_edgar_names, k=max_sec_edgar_names)
    data.extend(sec_edgar_names)

random.shuffle(data)
data = f"!{'!'.join(data)}!"

print(f"length of dataset in characters: {len(data):,}")

# get the vocabulary
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: "".join([itos[i] for i in l])

# create the train and validation splits
n = len(data)
train_data = data[: int(n * split_frac)]
val_data = data[int(n * split_frac) :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
