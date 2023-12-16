import argparse
import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

#
import torch.distributed as dist

batch_size = 64  # how many independent sequences will we process in parallel
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
#
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
#
torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.05  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageMmodel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_talbe = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_talbe(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # softmax and loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies in the current context
        for _ in range(max_new_tokens):
            # crop ind to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # become (B, C)
            # apply softmax to get probablities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


#
def get_batch(split, train_data, val_data):
    # generate a small batch of data of input x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# train
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument(
        "--model-type",
        type=str,
        default="custom",
    )
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    return parser.parse_args()


def get_data():
    # prepare training data
    with open("/opt/ml/input/data/training/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    # char set
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # mapping
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    # encode and decode
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text))
    # split train and val_data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[:n]
    return train_data, val_data, vocab_size, decode


def load_model():
    model = torch.load("./model/nanoGPT.pt")
    print(model)


def train():
    args = parse_args()
    # get data from s3
    train_data, val_data, vocab_size, decode = get_data()
    # init process group
    world_size = len(args.hosts)
    host_rank = args.hosts.index(args.current_host)
    print(f"host rank is {host_rank}")
    dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
    # device
    device = "cuda"
    # model
    model = BigramLanguageMmodel(vocab_size=vocab_size)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    #
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #
    # batch_size = 32
    for iter in range(max_iters):
        # sample a batch of data
        xb, yb = get_batch("train", train_data, val_data)
        # evaluate the loss
        logits, loss = model(xb, yb)
        # optimizer step
        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        # for distributed training
        loss.mean().backward()
        optimizer.step()
        if iter % eval_interval == 0:
            print(f"train loss {loss} and average {loss.mean().item()}")
    # save model
    try:
        torch.save(model.cpu().state_dict(), "/opt/ml/model/nanoGPT.pth")
    except:
        print('not able to save model')
    # generate not work in distributed mode
    # print(
    #     decode(
    #         model.generate(
    #             idx=torch.zeros((1, 1), dtype=torch.long, device=device),
    #             max_new_tokens=500,
    #         )[0].tolist()
    #     )
    # )
    return None


if __name__ == "__main__":
    train()
    # load_model()
