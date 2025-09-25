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
The paper creates a distributed representation as a vector embedding of words based on its context, which allows different words that are semantically similar to be rated as similar each other. The model will then use these embeddings into a neural network to compute the probability of the next word (by maximizing log-likelihood of both outcomes). They use an evaluation called perplexity to measure the model, and their model with both the n-gram and NN performs significantly better than any known n-gram models at that time. Although their results help with reducing the curse of dimensionality, the computational power increased greatly as well. 

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


### Download

The tiny config uses the Brown dataset, configurable in the download.py file. 
- Run `python -m nplm.download --dataset {brown,wikitext2} [--out_dir OUT_DIR]`
- Example: `python -m nplm.download --dataset brown`
- I enforce a specific dataset choice, which is either brown or wikitext2 (for the medium config). The output directory is optional, if not specified it is placed in `data/raw/<dataset>`

- For preproccessing, run `preprocess.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR [--shard_size SHARD_SIZE] [--lowercase] [--test_prop TEST_PROP] [--validation] [--val_prop VAL_PROP]`
- Example: `python -m nplm.preprocess --input_dir data/raw/brown --output_dir data/brown_jsonl --shard_size 10000 --lowercase`
- While input and output directories are mandatory, others have a default. If there is already data in the output directory, it is overwritten.

### Training and Evaluation
quickstart commands to install, download data, preprocess, train, and evaluate; 

### CPU/GPU usage notes
CPU/GPU usage notes (which did you use; what hardware should the instructor have at hand to reproduce your results)
