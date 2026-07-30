"""BPR training loop shared by MF and LightGCN, with periodic full-ranking
evaluation, early stopping on Recall@20, and result/log persistence."""

import json
from pathlib import Path

import numpy as np
import torch

from src.eval.evaluator import evaluate
from src.utils.common import ROOT, Timer, append_result


def train_bpr(model, data, cfg: dict, device: str) -> dict:
    tc, run = cfg["train"], cfg["experiment_name"]
    log_dir = ROOT / "experiments" / "logs" / run
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    rng = np.random.default_rng(cfg["seed"])
    n_batches = data.R.nnz // cfg["data"]["batch_size"]

    def score_fn(user_ids):
        model.eval()
        with torch.no_grad():
            u = torch.as_tensor(user_ids, device=device)
            return model.full_scores(u).cpu().numpy()

    best, best_epoch, patience, history = None, -1, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for _ in range(n_batches):
            users, pos, neg = data.sample_bpr_batch(cfg["data"]["batch_size"], rng)
            users = torch.as_tensor(users, device=device)
            pos = torch.as_tensor(pos, device=device)
            neg = torch.as_tensor(neg, device=device)
            loss, reg = model.bpr_loss(users, pos, neg)
            total = loss + tc["l2_reg"] * reg
            opt.zero_grad()
            total.backward()
            opt.step()
            ep_loss += loss.item()
        print(f"epoch {epoch:4d} | bpr_loss {ep_loss / n_batches:.4f}", flush=True)

        if epoch % tc["eval_every"] == 0:
            with Timer(f"eval @ epoch {epoch}"):
                m = evaluate(score_fn, data, topks=cfg["eval"]["topk"])
            print("  " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            history.append({"epoch": epoch, **m})
            (log_dir / "history.json").write_text(json.dumps(history, indent=2))
            if best is None or m["recall@20"] > best["recall@20"]:
                best, best_epoch, patience = m, epoch, 0
                torch.save(model.state_dict(), log_dir / "best.pt")
            else:
                patience += 1
                if patience >= tc["early_stop_patience"]:
                    print(f"early stop at epoch {epoch} (best @ {best_epoch})")
                    break

    append_result(run, cfg["model"]["name"], best, epoch=best_epoch)
    print(f"BEST (epoch {best_epoch}): " + " ".join(f"{k}={v:.4f}" for k, v in best.items()))
    return best
