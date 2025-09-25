#!/usr/bin/env python
# nplm/download.py
"""
Downloader (required) Provide a CLI to fetch datasets (e.g., via Hugging Face datasets or direct URLs):
Downloads the dataset to the specified directory

Run with:
    python -m nplm.download --dataset brown --out_dir data/raw/brown

    
"""
import argparse
import os

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downloads specified dataset to output directory")
    p.add_argument(
        "--dataset",
        choices=["brown", "wikitext2"],
        required=True,
        help="Either wikitext2 or brown. Must be either of these two strings.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        help="Path to save the output directory. If not specified, it will be in data/raw/<dataset>",
    )

    return p.parse_args()

def download_brown(output_dir):
    if not os.listdir(output_dir):
        import nltk
        print("Downloading Brown dataset to", output_dir)
        nltk.download("brown", download_dir=output_dir)
    else:
        print("Brown corpus already downloaded into", output_dir)

def download_wiki(output_dir):
    # TODO: implement
    print("Not implemented yet")
    pass

def main():
    args = parse_args()
    dataset = args.dataset
    output_dir = 'data/raw/' + dataset if args.out_dir is None else args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    if dataset == 'brown': 
        download_brown(output_dir)
    else: 
        download_wiki(output_dir)
    print("Download complete! Data saved at:", output_dir)


if __name__ == "__main__":
    main()