# 07 — Sequence Models

**Status:** Core learning and implementation completed; final review/documentation pending.

Understanding how recurrent neural networks process ordered data and learn dependencies across time.

## Topics covered

- Sequence data and sliding-window construction
- Vanilla recurrent neural networks (RNNs)
- Hidden states and sequence outputs
- Vanishing and exploding gradients
- Backpropagation through time (BPTT)
- LSTM architecture: cell state, hidden state, forget/input/output gates
- GRU architecture: update/reset gates and hidden state
- LSTM vs GRU parameter-count comparison
- Sequence prediction with a recurrent model
- A small recurrent sentiment-classification demonstration

## Notebook

- [`main.ipynb`](./main.ipynb) — explanations, PyTorch implementations, experiments, and exercises.

## Scope boundary

Text preprocessing and language modelling are not taught comprehensively in this chapter. The sentiment-classification example demonstrates how a recurrent layer can be used in a downstream task.

The following topics are reserved for **Chapter 10 — Embeddings & Language Models**:

- Tokenization
- Vocabulary construction and token IDs
- Unknown tokens, padding, sequence lengths, and batching
- Embedding layers and embedding spaces
- Positional embeddings
- Language modelling and next-token prediction
- Context and transformer language models
- Training objectives and inference

Chapter 08 introduces attention; Chapter 09 focuses on transformer architecture; Chapter 10 develops representations and language-modelling objectives.

## Important caveat

The toy next-number and sentiment datasets are teaching examples. Their results are not evidence of robust real-world generalization. For forecasting, use chronological evaluation splits. For text classification, use sufficiently large and representative datasets with separate training, validation, and test sets.

## Suggested completion checklist

- [ ] Run every notebook cell from top to bottom
- [ ] Confirm the parameter-count comparison
- [ ] Review outputs and remove stale/error outputs if needed
- [ ] Update the repository root README with Chapter 07 status
- [ ] Commit and push the chapter
