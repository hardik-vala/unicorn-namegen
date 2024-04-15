import numpy as np
import os
import pickle
import torch

from model import ModelConfig, NameModel

# ------------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_iters = 200
log_interval = 200
# data
dataset = "names"
batch_size = 16
block_size = 12  # context length
# model
n_layer = 2
n_head = 2
n_embd = 32
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-4  # max learning rate
max_iters = 1000
# system
device = (
    "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
# ------------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}  # for logging
# ------------------------------------------------------------------------------

tokens_per_iter = batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(24)

# data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join("data", "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join("data", "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64))
            for i in ix
        ]
    )
    return x, y


# attempt to derive vocab_size from the dataset
meta_path = os.path.join("data", "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size,
    dropout=dropout,
)

modelconf = ModelConfig(**model_args)
model = NameModel(modelconf)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# training loop
for step in range(max_iters):
    if step % log_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# write checkpoint
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "model_args": model_args,
    "iter_num": step,
    "final_val_loss": losses["val"],
    "config": config,
}
print(f"saving checkpoint to {out_dir}")
torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))