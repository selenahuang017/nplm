# Implement a Neural Probabilistic Language Model (NPLM)


## Introduction
In this one-week, hands-on assignment, you will implement a **Neural Probabilistic Language Model** (NPLM) inspired by **Bengio et al., 2003**. Your system will:
- Download a public text corpus,
- Preprocess it into **JSONL shards** (one sentence, or paragraph, or "document" per line -- your choice),
- Build a **tokenizer** and **vocabulary** (either classic word-level or modern BPE),
- Train a **feed-forward** next-word predictor with a fixed context window,
- Evaluate and report **perplexity**. (If you don't know what this is yet, we will talk about it on Monday, Sep. 15, at the start of Module 4).

You are encouraged to leverage LLMs to accelerate development—*responsibly and transparently* (see “Responsible LLM Use”). Partly this is because -- for those who haven't yet practiced incorporating LLMs into their development and research cyle -- knowing how to do so judiciously is critical (or will soon be critical) for success in the modern workplace; a project that might take weeks can, if LLMs are properly leveraged, take days, and in the end be better organized, better documented, and be fitted with more thorough test suites. Those of us who are not accustomed to building LLM usage into our development cycle will not fare well when put up against this unprecedented level of productivity.

That being said, a couple of caveats are in order. Firstly, you're here to learn; make sure you understand the code you're organizing into your repository -- if writing something yourself seems necessary to support full understanding, then do so; if you find interrogating the LLM in pursuit of explanations, commentary, etc., helps with understanding, make sure you take the time to leverage this super-intelligent tutor to your best advantage. Secondly, using an LLM well is a skill like any other -- make sure you pay close attention to what is working, what isn't working, and adjust as necessary, all toward the end of building and cultivating this novel skill.


## Setup

1. **Clone this starter repo (the instructor’s copy):**
   ```bash
   git clone <URL-of-instructor-repo> nplm-assignment
   cd nplm-assignment
   ```

2. **Remove the link to the instructor’s remote** (so you don’t accidentally push there):
   ```bash
   git remote remove origin
   ```

3. **Create a new private repository** under your own GitHub/GitLab account.  
   - On GitHub: click the green “New” button, choose **Private**, and name it something like `nplm-assignment`.  
   - Don’t initialize with a README or .gitignore (we already have those here).

4. **Add your new repo as the remote**:
   ```bash
   git remote add origin git@github.com:<your-username>/<your-repo>.git
   ```

5. **Push the starter code into your repo**:
   ```bash
   git push -u origin main
   ```
   (If your local branch is `master` instead of `main`, replace `main` with `master`.)

6. **Install dependencies** in a fresh environment (Python 3.10+ recommended; also recommended to be in a virtual environment before you run this):
   ```bash
   pip install -e .
   ```
   - This uses the included `pyproject.toml` to install your package in editable mode. 
   - [Note: you may also include dependencies in a `requirements.txt` and run `pip install -r requirements.txt` if you prefer.]

7. **Check that it runs**:
   ```bash
   python -m nplm.test_install
   ```
   - **NB:** This command may take a little while to run for the first time. Also, don't worry if you see something like: `FutureWarning: Using TRANSFORMERS_CACHE is deprecated and will be removed in v5 of Transformers. Use HF_HOME instead.`

---

At this point you’ll have a **private repo under your control**, with the starter code installed and ready to extend. All of your commits and pushes should go to *your* private repo. Don’t forget to give the instructor read access when you submit.


## Deliverables
Submit a private Git repository (give the instructor access) containing:
- `README.md` (≤1 page): a 2–4 sentence summary of the Bengio paper’s core idea; quickstart commands to install, download data, preprocess, train, and evaluate; CPU/GPU usage notes (which did you use; what hardware should the instructor have at hand to reproduce your results)
  - There should be enough here for the instructor to quickly and easily install your repository and run your code, replicating your results
- Code under a package (suggested: `nplm/`).
- At least two configs in `configs/` (e.g., `tiny.yaml`, `medium.yaml`).
  - The `tiny.yaml` should represent an easily reproducible **CPU-runnable training configuration** (<10 min on typical CPU).
- `results/metrics.json` (final validation & test perplexity, training time, vocab size, tokenizer type) and `results/EXPERIMENTS.md` (≤¾ page: dataset, preprocessing choices, model sizes tried, what helped/hurt).
  - If you prefer to put your experimental setup notes in `README.md`, you may; sometimes, though, it helps to separate this out in a separate document
- `LLM_LOG.md` (≤1 page): how you used LLMs, with prompts, what you changed/verified, and one concrete issue you caught -- more on this towards the end of this document.

A suggested repo layout appears in “Tips/Gotchas”.

---


## Corpus Selection and Preprocessing
You may choose from the menu below or select a comparable open corpus.

**Recommended options**
- **Tiny**: Shakespeare's works
- **Brown**: Used in the paper, can be had easily by downloading through NLTK (documentation easily googlable)
- **Medium**: WikiText-2 (`wikitext-2`)
- **Stretch (GPU)**: WikiText-103 (`wikitext-103`)

**Downloader (required)**
Provide a CLI to fetch datasets (e.g., via Hugging Face `datasets` or direct URLs):

```bash
python -m nplm.download --dataset wikitext2 --out_dir data/raw/wikitext2
```

**Preprocessing (required)**
Write a script that:
1) Reads raw data,
2) Splits into **documents** (your choice — paragraphs, sentences, wiki articles, etc.; be explicit, and feel free to err towards whatever divisions the raw corpus seems to make most natural),
3) Generates a train/test split, or train/val/test split, with controllable proportions (supplied as CLI arguments)
4) Writes **JSONL shards** to `data/<corpus-name>/<train\test\split>/shard_<n>.jsonl` with `--shard_size` docs per file.

**JSONL schema**
Each line is one JSON object with at least:
```json
{"text": "document text here"}
```
You may include extra metadata fields (e.g., `{"id": "...", "source": "...", "text": "..."}"`).

**Example CLI**
```bash
python -m nplm.preprocess \
  --input_dir data/raw/wikitext2 \
  --output_dir data/wikitext2_jsonl \
  --shard_size 10000 \
  --lowercase
```

---


## Tokenization
You must support **one** of the two paths below; supporting **both** is encouraged.

### Option A — Word-Level (Bengio-style)
**NB:**
We have already provided you with a ready-implementation of a word-level tokenizer. Feel free to use it. Also feel free to write your own, if you'd like the experience -- it's not a hard task.

If you choose to write your own, note that your tokenizer module will:
- Build a **closed vocabulary** from the training split (frequency cutoffs/merges as described in the paper are allowed)
- Add to the vocabulary required special tokens: `<pad>`, `<unk>`; optionally `<bos>`, `<eos>`.
- Map OOV words to `<unk>`.
- Accept a directory of JSONL shards and output an artifact bundle (e.g., a `.json` vocab file and any auxiliary files). Provide a CLI like:
```bash
# Build vocab from train shards
python -m nplm.build_word_vocab \
  --jsonl_dir data/wikitext2_jsonl/train \
  --output_path artifacts/wikitext2_word_vocab.json \
  --min_freq 2 \
  --max_vocab 20000

# Encode JSONL shards to token IDs (if you choose to pre-encode)
python -m nplm.encode_word \
  --jsonl_dir data/wikitext2_jsonl \
  --vocab artifacts/wikitext2_word_vocab.json \
  --output_dir data/wikitext2_encoded
```

**Runtime encoding is also acceptable** (i.e., build vocab once, then encode on the fly during training).

### Option B — Modern Sub-Word Tokenizer
We don't use closed-vocabulary word-level tokenizers anymore. Instead, we use sub-word tokenizers, usually trained on subsets of our model-training corpus using something like SentencePiece. 
- You may use a standard modern tokenizer -- and if you do so, I recommend using something like the Llama 3 tokenizer, which can be easily had by loading it from the HuggingFace model hub using the `transformers` library, e.g.:
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
```
This is a gated model, so you'll have to submit a request for access on [this page](https://huggingface.co/meta-llama/Llama-3.1-8B), but it's usually a trivially short wait, and the tokenizer is pretty high quality. Then, before running your code, you'll have to make to run:
```
huggingface-cli login
```
in your terminal, and supply a key you generate on your HuggingFace account page, [here](https://huggingface.co/settings/tokens).

Or you could train your own from scratch. If you do so, be explicit about the **training corpus** used for learning merges, the vocab size, and any other architectural details that seem relevant.

**Tokenizer builder interface (if training your own BPE):**
```bash
python -m nplm.build_bpe \
  --jsonl_dir data/wikitext2_jsonl/train \
  --output_dir artifacts/wikitext2_bpe \
  --vocab_size 16000
```

### Final note:
Just remember, while **using a modern sub-word tokenizer is allowed** (e.g., `gpt2`, `llama`-family), this **does** change the modeling granularity relative to Bengio’s word-level formulation. You'll get different results. They might (probably will?) be better! **Particularly** if you train your own, and use a relatively small vocabulary. That’s fine—just document and compare.

---


## Training
Implement a **feed-forward** NPLM with a fixed **context window** of size `n` (e.g., `n=5` previous tokens predict next token).

**Required architecture**
1. Embedding lookup for each of the `n` context tokens (shared matrix).  
2. Concatenate embeddings → hidden layer (e.g., `tanh` as in paper, or `ReLU`, or modern variant; optional dropout, but again, note that this wasn't used in the Bengio paper).
3. Linear → softmax over the vocabulary.
4. Train with cross-entropy; report perplexity.

**Data handling**
- Build training examples by sliding a window within each **document** independently. Do cross document boundaries unless you explicitly choose to and document it.
- For left boundary, you may pad with `<pad>` or use `<bos>`; this is a meaningless but necessary choice.

**Configs & CLI (required)**
As noted before, provide at least a tiny and a medium config. Example fields (your code may or may not use all of these):
```yaml
seed: 1337
tokenizer_type: "word"   # or "bpe"
tokenizer_path: "artifacts/wikitext2_word_vocab.json"
context_size: 5
vocab_size: 20000        # ignored if derived from tokenizer artifact
embedding_dim: 128
hidden_dim: 256
dropout: 0.1
optimizer: "adam"
lr: 0.001
batch_size: 256
max_steps: 20000
eval_every: 1000
clip_grad_norm: 1.0
device: "auto"           # "cpu", "cuda", or "auto"
```

**Example commands**
```bash
# Train
python -m nplm.train \
  --config configs/tiny.yaml \
  --data_dir data/wikitext2_jsonl \
  --save_dir runs/exp1

# Resume / change config
python -m nplm.train \
  --config configs/medium.yaml \
  --data_dir data/wikitext2_jsonl \
  --save_dir runs/exp2 --resume_from runs/exp1/best.pt
```

**Efficiency**
- For vocab ≤ 20–50k, full softmax is fine. Larger vocabs may benefit from sampled/adaptive softmax (optional for extra credit! just document that you did this, and document results).
- Provide at least one config that trains to completion in **<10 minutes on CPU** for grading.

---


## Evaluation
Report **train** and **test** perplexity for best checkpoint and, if you feel so inclined, log training time.

**Perplexity**
Compute perplexity (this is `exp(mean_token_level_loss)`) on the evaluation split.

**CLI**
```bash
python -m nplm.eval \
  --checkpoint runs/exp1/best.pt \
  --data_dir data/wikitext2_jsonl \
  --out_json results/metrics.json
```

**NB**: you could simply have your training script output the `metrics.json` file at the end of the training run; your choice.

`results/metrics.json` should include: `{ "train_ppl": ..., "test_ppl": ..., "tokenizer": "...", "vocab_size_or_merges": ..., "context_size": ..., "train_time_sec": ... }`.

Include a short `results/EXPERIMENTS.md` describing dataset choice, preprocessing, tokenizer choice, configs tried, and any observations you care to make.

---


## Responsible LLM Use
You **must** include `LLM_LOG.md` with:
- Which LLM(s) you used,
- Some description of **how** you used them, and how helpful you found them

Optionally, you might also include:
- Representative prompts and outputs,
- How you verified and modified them,
- One concrete bug, misconception, or performance issue you caught and how you fixed it -- if such transpires

---


## Tips/Gotchas
**Suggested repo structure**
```
nplm/
  __init__.py
  download.py
  preprocess.py
  word_tokenizer.py          # Option A tokenizer (word)
  build_word_tokenizer.py    # Option A script (applies option A tokenizer)
  encode_word.py             # Option A (optional, if pre-encoding corpus)
  build_subword_tokenizer.py # Option B (optional, if training own subword tokenizer from scratch, rather than using a stock one, like Llama 3)
  encode_subword.py          # Option B (optional, if pre-encode corpus)
  data.py                    # Represents JSONL dataset, on-the-fly windowing, used by training script as "dataloader"
  model.py                   # NPLM model: embeddings -> hidden -> softmax
  train.py
  eval.py
  utils.py                   # logging, seeds, checkpointing, perplexity, timers
configs/
  tiny.yaml
  medium.yaml
results/
  metrics.json
  EXPERIMENTS.md
runs/                     # checkpoints & logs (gitignored)
README.md
LLM_LOG.md
```

**Random design notes** -- ignore if not helpful:
- **Word-level vs BPE**: different token granularities change perplexity scale and softmax cost -- document choices and compare if you try both.
- **Vocab size** (word-level): start ~10–20k; `<unk>` rate matters -- log it if you can
- **Context size**: try `n ∈ {3, 5, 7}`; larger `n` increases input dimensionality (concat embeddings)
- **Regularization**: dropout on hidden; gradient clipping helps stability, though you'll likely not need it at this scale
- **Repro**: fix seeds (`torch`, `numpy`, Python) and log them -- in fact, perhaps include them in your config files; indicate hardware

---

## Extra Considerations (Optional + Credit)
- Sampled/adaptive softmax for large vocabs,
- Tied input–output embeddings,
- Compare word-level vs BPE,
- Plot train/val perplexity over steps and include the PNG in `results/`.

---

### Grading (summary)
- Repo clarity & README (15)
- Data pipeline & JSONL sharding (15)
- Tokenizer/vocab implementation & interface (15)
- Model correctness (feed-forward, fixed window) (25)
- Training & evaluation results (20)
- LLM collaboration log (10)
