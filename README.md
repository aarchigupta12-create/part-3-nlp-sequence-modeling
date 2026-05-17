# part-3-nlp-sequence-modeling

# Part 3: NLP and Sequence Modeling Mini Project

This repository contains an end-to-end NLP framework executing data pipeline preparation, traditional TF-IDF feature extraction, structural baseline modeling, and sequence-based recurrent neural network training using real-world user communication context.

---

## Task 1 & 2: Dataset & Preprocessing Insight

### Dataset Metrics
Based on the provided `customer_support_text_classification.csv` data:
* **Total Records:** 1,496
* **Target Labels:** `neutral`, `positive`, `negative`
* **Average Document Length:** ~12-14 words per ticket.

### Preprocessing Protocol
Text blocks are fed through a systematic parsing pipeline:
1. **Lowercasing:** Eliminates duplicate vocabulary state representations caused by capitalization variations.
2. **Regex Cleansing:** Strips punctuations and structural noise using `[^a-z0-9\s]` matching.
3. **Tokenization:** Breaks contiguous textual sequences into singular whitespace-delimited sub-strings.
4. **Sequence Stabilization:** Sequence lengths are universally aligned to `max_len=30` using post-padding structures to generate uniform inputs for the tensor arrays.

---

## Task 3: Vectorization Justification

### Why Text Must Be Converted Into Vectors
Mathematical systems and algorithmic models (such as neural networks or linear classifiers) inherently perform matrix computations using weights and biases; they cannot process structural string properties or human syntax natively. Vectorization bridges this gap by mapping qualitative human dialect into deterministic multi-dimensional metric spaces, turning language tokens into numerical representations while preserving syntactic or statistical semantics.

---

## Task 5: Sequence Model Architectural Blueprint

The workflow for processing incoming sequence signals inside our implemented LSTM deep network handles dimensions as follows:

```text
[Input String] 
      │ (Length: 30 Tokens)
      ▼
[Embedding Layer] ────► Converts token index to 64-dimensional dense space vector
      │
      ▼
[LSTM Layer (64 Units)] ──► Progressively steps through the 30 tokens, maintaining an internal hidden memory state
      │
      ▼
[Dense Layer (ReLU)] ──► Distills the final sequence representation into 32 deep latent features
      │
      ▼
[Output Layer (Softmax)] ──► Produces a probability array across the 3 target classes
