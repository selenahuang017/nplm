#!/usr/bin/env python
# nplm/preprocess.py
"""
Preprocessing pipeline, to 
    1. read the raw data from a specified folder
    2. split into documents of choice (I chose sentences)
    3. generate a train/test split, or include validation if so desired
    4. writes to JSONL shards in data/<corpus>/<train/test/split>/shard_<n>.jsonl
    Note: the JSONL schema should have at least the "text" field.

Run with:
    python -m nplm.preprocess --input_dir data/raw/brown \
        --output_dir data/brown_jsonl --shard_size 10000 --lowercase

"""
import argparse
import os, sys
import json
from sklearn.model_selection import train_test_split

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocessing pipeline to create JSONL shards")
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="input directory, where the data is stored",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the shards.",
    )
    p.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Shard size. Default is 10000.",
    )
    p.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before tokenization.",
    )
    p.add_argument(
        "--test_prop",
        type=float,
        default=0.2,
        help="Test dataset proportion, defaults to 0.2.",
    )
    p.add_argument(
        "--validation",
        action="store_true",
        help="Add validation set flag.",
    )
    p.add_argument(
        "--val_prop",
        type=float,
        default=0.1,
        help="Validation dataset proportion, defaults to 0.1.",
    )

    return p.parse_args()

def download_brown(input_dir):
    from nltk.corpus import brown
    import nltk.data

    nltk.data.path.append(input_dir)
    return [" ".join(sent) for sent in brown.sents()]

def download_wiki(input_dir):
    pass

def save_dataset(output_dir, dataset, dset_name, shard_size):
    # writes to JSONL shards in data/<corpus>/<train/test/split>/shard_<n>.jsonl
    path = output_dir + '/' + dset_name
    os.makedirs(path, exist_ok=True)
    print(f"Saving dataset {dset_name}...")

    # this loop was created with edited ChatGPT output because i deadass didn't know how to create jsonl files
    for i in range(0, len(dataset), shard_size):
        chunk = dataset[i:i+shard_size]
        filename = os.path.join(path, f"shard_{i//shard_size+1}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for text in chunk:
                json.dump({"text": text}, f)
                f.write("\n")


def main():
    args = parse_args()

    # read data and split into documents
    if not os.listdir(args.input_dir):
        print(f"Error: No data in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)
    data = download_brown(args.input_dir) if "brown" in args.input_dir else download_wiki(args.input_dir)

    # lowercase 
    if args.lowercase: 
        data = [sent.lower() for sent in data]
    
    # split dataset
    data_train, data_test = train_test_split(data, test_size=args.test_prop, random_state=42)
    data_val = None
    if args.validation: 
        data_train, data_val = train_test_split(data_train, test_size=args.val_prop, random_state=42)

    save_dataset(args.output_dir, data_train, 'train', args.shard_size)
    save_dataset(args.output_dir, data_test, 'test', args.shard_size)
    if args.validation: 
        save_dataset(args.output_dir, data_val, 'val', args.shard_size)
    print("Preprocessing complete! Data saved at:", args.output_dir)


if __name__ == "__main__":
    main()