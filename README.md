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

## Installation

Clone the repo and install in **editable mode**:

```bash
git clone <your-assignment-repo>
cd <your-assignment-repo>
pip install -e .
```

This uses the `pyproject.toml` provided in the repo.

- [Note: you may place dependencies directly under `[project] dependencies` in `pyproject.toml`, or keep them in a separate `requirements.txt` file and update accordingly.]

---

## Next Steps

See **ASSIGNMENT.md** for full instructions on how to use and extend this scaffold.
