import os
import pickle
import torch
from model import ModelConfig, NameModel

# -----------------------------------------------------------------------------
out_dir = "out"  # ignored if init_from is not 'resume'
num_samples = 10  # number of samples to draw
max_new_tokens = 100  # number of tokens generated in each sample
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 24
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# -----------------------------------------------------------------------------

torch.manual_seed(seed)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
modelconf = ModelConfig(**checkpoint["model_args"])
model = NameModel(modelconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()

meta_path = os.path.join("data", "meta.pkl")
print(f"Loading meta from {meta_path}...")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

with torch.no_grad():
    for k in range(num_samples):
        x = torch.full((1, 1), stoi["!"], dtype=torch.long)
        y = model.generate(x, max_new_tokens)
        raw = decode(y[0].tolist())
        parts = raw.split("!")
        i = torch.randint(low=1, high=len(parts) - 2, size=(1, 1), dtype=torch.long).item()
        print(parts[i])
        print("---------------")
