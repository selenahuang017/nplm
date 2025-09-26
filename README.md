# Neural Probabilistic Language Model (NPLM)

This repository is a starting point for the one-week assignment based on
Bengio et al. (2003), *A Neural Probabilistic Language Model*.

Students will extend this scaffold into a working project that can:
- Download and preprocess a text corpus into JSONL shards,
- Tokenize the text (either **word-level** or **BPE**),
- Train a feed-forward neural language model with a fixed context window,
- Evaluate perplexity on validation/test data,
- Document the process, results, and their collaboration with LLMs.

---
## Assignment README

### Bengio paper summary
The paper creates a distributed representation as a vector embedding of words based on its context, which allows different words that are semantically similar to be rated as similar each other. The model will then use these embeddings into a neural network to compute the probability of the next word (by maximizing log-likelihood of both outcomes). They use an evaluation called perplexity to measure the model, and their model with both the n-gram and NN performs significantly better than any known n-gram models at that time. Although their results help with reducing the curse of dimensionality, the computational power required increased greatly as well. 

---
## Notes
These are notes I had while doing each section, a shortened list of commands are at the bottom. 

### Installing this repo
Clone the repo and install in **editable mode**:

```bash
git clone <your-assignment-repo>
cd <your-assignment-repo>
pip install -e .
```

### Setup 
- After creating a virtual environment, you can install everything using `pip install .`
- Run `python -m nplm.test_install` from the base folder to check if all installs were done correctly. 


### Download and Preprocessing

The tiny config uses the Brown dataset, configurable in the download.py file. 
- Run `python -m nplm.download --dataset {brown,wikitext2} [--out_dir OUT_DIR]`
- Example: `python -m nplm.download --dataset brown`
- I enforce a specific dataset choice, which is either brown or wikitext2. Though originally i wanted to be able to support both, I ran out of time to implement the read of wikitext2, so it is actually currently unimplemented. The tiny and medium configs work on the Brown corpus. The output directory is optional, if not specified it is placed in `data/raw/<dataset>`

- For preproccessing, run `preprocess.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR [--shard_size SHARD_SIZE] [--lowercase] [--test_prop TEST_PROP] [--validation] [--val_prop VAL_PROP]`
- Example: `python -m nplm.preprocess --input_dir data/raw/brown --output_dir data/brown_jsonl --shard_size 10000 --lowercase`
- While input and output directories are mandatory, others have a default. If there is already data in the output directory, it is overwritten.

### Tokenization

Though this seems to be an interesting task, I do know that there are multiple libraries (like nltk) that have a number of different ways of tokenizing sentences, so I will just use the one provided. 

There is slight confusion in the assignment: The file provided is `build_word_vocab.py` but in the suggested repo structure it expects a `build_word_tokenizer.py` so I just renamed the file, but it is using the given tokenizer. 
- Run example: `python -m nplm.build_word_tokenizer --jsonl_dir data/brown_jsonl/train --output_path artifacts/brown_word_vocab.json --lowercase`

There is no encode_word.py

### Training and Evaluation
Training: 
- I used the tiny.yml set to do training, the medium.yml is just the provided 
- Usage: train.py --data_dir DATA_DIR --save_dir SAVE_DIR --config CONFIG [--resume-from RESUME_FROM]
- Example: `python -m nplm.train --config configs/tiny.yaml --data_dir data/brown_jsonl --save_dir runs/exp1`
quickstart commands to install, download data, preprocess, train, and evaluate; 

Evaluation:
- Usage: eval.py --checkpoint CHECKPOINT --data_dir DATA_DIR --out_json OUT_JSON
- This follows the process outlined in the assignment.
- Example: python -m nplm.eval --checkpoint runs/exp2/best.pt --data_dir data/brown_jsonl --out_json results/metrics.json
- `python -m nplm.eval --checkpoint runs/exp1/best.pt --data_dir data/brown_jsonl --out_json results/metrics.json`
- The output is as expected, in results/metrics.json

I didn't actually use a util.py

---
## Run commands
- `python -m nplm.download --dataset brown`
- `python -m nplm.preprocess --input_dir data/raw/brown --output_dir data/brown_jsonl --shard_size 5000 --lowercase`
- `python -m nplm.build_word_tokenizer --jsonl_dir data/brown_jsonl/train --output_path artifacts/brown_word_vocab.json --lowercase`
- `python -m nplm.train --config configs/tiny.yaml --data_dir data/brown_jsonl --save_dir runs/exp1`
- `python -m nplm.eval --checkpoint runs/exp1/best.pt --data_dir data/brown_jsonl --out_json results/metrics.json`

### CPU/GPU usage notes
I didn't use a GPU, just CPU on my macbook so a standard M3 Base model mac (yes sorry for my potato computer)
- CPU: 8-core (4 performance cores + 4 efficiency cores)
- Neural Engine: 16-core
- Memory Bandwidth: 100 GB/s
- Clock Speed: Up to 4.05 GHz