from contextlib import nullcontext
import os
import pickle
import torch
from model import ModelConfig, Namegen

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
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ctx = nullcontext()


def sample_names(num_samples):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    modelconf = ModelConfig(**checkpoint["model_args"])
    model = Namegen(modelconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    meta_path = os.path.join("data", "meta.pkl")
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    merges = meta["merges"]
    stoi, itos = meta["stoi"], meta["itos"]
    decode1 = lambda l: "".join([itos[i] for i in l])

    def unmerge(ids, pair, idx):
        newids = []
        for i in ids:
            if i == idx:
                newids.append(pair[0])
                newids.append(pair[1])
            else:
                newids.append(i)
        return newids

    def decode(ids):
        tokens = list(ids)
        for pair, idx in reversed(merges.items()):
            tokens = unmerge(tokens, pair, idx)
        return decode1(tokens)

    names = []
    with torch.no_grad():
        while True:
            x = torch.full((1, 1), stoi["!"], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens)
            raw = decode(y[0].tolist())
            parts = raw.split("!")
            for i in range(1, len(parts) - 1):
                names.append(parts[i])
                if len(names) >= num_samples:
                    break
            if len(names) >= num_samples:
                break
    
    return names


def main():
    names = sample_names(num_samples)
    for n in names:
        print("---------------")
        print(n)


if __name__ == "__main__":
    main()
