"""SASRec training loop: mirrors bpr_trainer's protocol (periodic full-ranking
evaluation, early stopping on Recall@20, config/history/checkpoint/result
persistence) so all models share one experimental discipline."""

import json

import numpy as np
import torch

from src.eval.evaluator import evaluate
from src.utils.common import ROOT, Timer, append_result, rng_restore, rng_snapshot


def train_sasrec(model, seq_data, data, cfg: dict, device: str,
                 resume: bool = False) -> dict:
    tc, run = cfg["train"], cfg["experiment_name"]
    log_dir = ROOT / "experiments" / "logs" / run
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                           weight_decay=tc.get("weight_decay", 0.0))
    rng = np.random.default_rng(cfg["seed"])

    def score_fn(user_ids):
        model.eval()
        with torch.no_grad():
            inp, length = seq_data.eval_inputs(np.asarray(user_ids))
            inp = torch.as_tensor(inp, device=device)
            length = torch.as_tensor(length, device=device)
            return model.full_scores(inp, length).cpu().numpy()

    best, best_epoch, patience, history, start_epoch = None, -1, 0, [], 1
    ckpt_path = log_dir / "last.ckpt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best, best_epoch = ck["best"], ck["best_epoch"]
        patience, history = ck["patience"], ck["history"]
        start_epoch = ck["epoch"] + 1
        if "rng" in ck:
            rng_restore(ck["rng"], rng)
        else:
            print("WARNING: checkpoint predates RNG-state persistence -- "
                  "resume will be statistically continued, not bit-exact")
        print(f"resumed from epoch {ck['epoch']}")
    elif resume:
        print("no checkpoint found, starting fresh")

    done_path = log_dir / "DONE"
    if resume and done_path.exists() and best is not None:
        print(f"run already completed (best recall@20={best['recall@20']:.4f} "
              f"@ epoch {best_epoch}); skipping")
        return best

    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        ep_loss, n_b = 0.0, 0
        for inp, tgt, neg, _ in seq_data.train_batches(
            cfg["data"]["batch_size"], rng
        ):
            inp = torch.as_tensor(inp, device=device)
            tgt = torch.as_tensor(tgt, device=device)
            neg = torch.as_tensor(neg, device=device)
            loss = model.training_loss(inp, tgt, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        print(f"epoch {epoch:4d} | bce_loss {ep_loss / n_b:.4f}", flush=True)

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
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": epoch, "best": best, "best_epoch": best_epoch,
                 "patience": patience, "history": history, "rng": rng_snapshot(rng)},
                ckpt_path,
            )
            if patience >= tc["early_stop_patience"]:
                print(f"early stop at epoch {epoch} (best @ {best_epoch})")
                break

    done_path.write_text(f"best_epoch={best_epoch}\n")
    append_result(run, "sasrec", best, epoch=best_epoch,
                  notes=f"max_len={cfg['model']['max_len']}", cfg=cfg)
    print(f"BEST (epoch {best_epoch}): "
          + " ".join(f"{k}={v:.4f}" for k, v in best.items()))
    return best
