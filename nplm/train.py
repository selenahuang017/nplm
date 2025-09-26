import argparse
import os
import yaml
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

from .model import NPLM
from .data import create_dataloader

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training the model")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="data directory with jsonl shards",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save the training output.",
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Location of the config.yaml file.",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        help="Load from a specific point.",
    )
    return p.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)

            log_probs = model(context)
            loss = criterion(log_probs, target)

            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)

    return total_loss / total_tokens

def train_full(train_loader, val_loader, cfg, args):
    step = 0
    # model defaults
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    model = NPLM(
        vocab_size=train_loader.dataset.vocab_size,
        context_size=cfg["context_size"],
        embed_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    if cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    start_epoch = 0
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    # Resume if checkpoint is given
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed from {args.resume_from}, starting at epoch {start_epoch}")

    start_time = time.time()
    
    for epoch in range(start_epoch + 1, cfg.get("epochs", 100) + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            log_probs = model(context)  # [batch, vocab_size]
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)
            
            if step % cfg["eval_every"] == 0 or step >= cfg["max_steps"]:
                val_loss = evaluate_perplexity(model, val_loader, criterion, device)
                
                # Save model
                total_time = time.time() - start_time
                log = {
                    "epoch": epoch,
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_args": {
                        "vocab_size": train_loader.dataset.vocab_size,
                        "context_size": cfg["context_size"],
                        "embed_dim": cfg["embedding_dim"],
                        "hidden_dim": cfg["hidden_dim"]
                    },
                    "train_time_sec": total_time,
                    "val_loss": val_loss,
                    "perplexity": math.exp(val_loss),
                    "tokenizer_path": cfg["tokenizer_path"]
                }
                torch.save(log, last_path)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(log, best_path)
                    print(f"Saved new best checkpoint to {best_path}")

                print(f"Step {step} | Time: {int(total_time // 60)} min {int(total_time % 60)} sec | Val loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")
                if step >= cfg["max_steps"]:
                    print("Reached max_steps. Stopping training.")
                    return
            step += 1

        train_loss = total_loss / total_tokens
        val_loss = evaluate_perplexity(model, val_loader, criterion, device)
        print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(cfg["seed"])

    train_loader = create_dataloader(
        folder=args.data_dir, split="train", vocab_path=cfg["tokenizer_path"],
        context_size=cfg["context_size"], batch_size=cfg["batch_size"], shuffle=True
    )
    if cfg.get("validation", False): # needed for perplexity evaluation
        val_loader = create_dataloader(
            folder=args.data_dir, split="val", vocab_path=cfg["tokenizer_path"],
        context_size=cfg["context_size"], batch_size=cfg["batch_size"], shuffle=False
        )
    else: 
        val_loader = train_loader

    train_full(train_loader, val_loader, cfg, args)

    print("Training finished.")


if __name__ == "__main__":
    main()
