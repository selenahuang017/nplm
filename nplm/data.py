# created with help from ChatGPT
import json
import torch
import os
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, folder, dset_name, vocab_path, context_size):
        """
        Args:
            folder: Path to folder containing shards (e.g., folder/train/*.jsonl)
            dset_name: training or validation load
            vocab_path: Path to vocab.json (must contain token_to_id, id_to_token)
            context_size: Number of previous tokens for prediction
        """
        self.context_size = context_size
        with open(vocab_path, "r") as f:
            vocab_obj = json.load(f)
        self.token_to_id = vocab_obj["token_to_id"]
        self.id_to_token = vocab_obj["id_to_token"]
        self.vocab_size = len(self.id_to_token)
        self.boundary_token = self.token_to_id.get("<bos>", 0)
        self.unk_id = self.token_to_id.get("<unk>", 0)

        split_dir = os.path.join(folder, dset_name)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Expected subfolder {split_dir} for split='{dset_name}'")
        self.files = [
            os.path.join(split_dir, shard)
            for shard in sorted(os.listdir(split_dir))
            if shard.endswith(".jsonl")
        ]

        self.samples = []
        for path in self.files:
            with open(path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj["text"].strip().split()
                    ids = [self.token_to_id.get(tok, self.unk_id) for tok in text]

                    # Pad left boundary with <bos> or <pad>
                    padded = [self.boundary_token] * context_size + ids

                    # Sliding window within this document only
                    for i in range(context_size, len(padded)):
                        context = padded[i - context_size : i]
                        target = padded[i]
                        self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def create_dataloader(folder, split, vocab_path, context_size, batch_size, shuffle=True):
    dataset = Data(folder, split, vocab_path, context_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)