import argparse
import time
import os
import math
import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)

def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16) 
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    
    # Fast BPE Tokenizer training checkpoint
    tok_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Training fast BPE tokenizer on a chunk of the corpus...")
        tok = tokenizer_mod.BPETokenizer()
        tok.train(text[:1000000], vocab_size=1024)
        tok.save(tok_path)
    
    tok = tokenizer_mod.load(tok_path)
    print("Encoding corpus...")
    ids_list = tok.encode(text)
    ids = torch.tensor(ids_list, dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens (vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    losses = []
    
    # Simulating batch size 64 to push max data through 2,000 optimization steps.
    grad_acc_steps = 4 
    
    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.steps, args.lr, args.lr * 0.1, warmup_steps=200)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
            
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        
        for _ in range(grad_acc_steps):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            loss = loss / grad_acc_steps
            loss.backward()
            step_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(step_loss)
        
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  ({(time.time()-t0)/step*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
