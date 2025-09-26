import argparse
import os
import json
import math
import torch
import torch.nn as nn

from .model import NPLM
from .data import create_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained NPLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing JSONL dataset")
    parser.add_argument("--out_json", type=str, required=True, help="Path to save metrics JSON")
    return parser.parse_args()


def evaluate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)

            logits = model(context)
            loss = criterion(logits, target)
            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss)


def main():
    args = parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["model_args"]
    
    device = model_args.get("device", "cpu")
    model = NPLM(
        vocab_size=model_args["vocab_size"],
        context_size=model_args["context_size"],
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    
    # Load criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloaders
    train_loader = create_dataloader(
        folder=args.data_dir, split="train",
        vocab_path=checkpoint["tokenizer_path"],
        context_size=model_args["context_size"],
        batch_size=model_args.get("batch_size", 256),
        shuffle=False
    )
    test_loader = create_dataloader(
        folder=args.data_dir, split="test",
        vocab_path=checkpoint["tokenizer_path"],
        context_size=model_args["context_size"],
        batch_size=model_args.get("batch_size", 256),
        shuffle=False
    )

    # Evaluate perplexity
    train_loss, train_ppl = evaluate_perplexity(model, train_loader, criterion, device)
    test_loss, test_ppl = evaluate_perplexity(model, test_loader, criterion, device)

    metrics = {
        "train_ppl": train_ppl,
        "test_ppl": test_ppl,
        "tokenizer": checkpoint["tokenizer_path"],
        "vocab_size_or_merges": model_args["vocab_size"],
        "context_size": model_args["context_size"],
        "train_time_sec": checkpoint.get("train_time_sec", None)
    }

    # Save JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation complete. Metrics saved to {args.out_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
