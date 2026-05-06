# Attention Mechanisms in PyTorch

This repository contains simple PyTorch implementations of attention mechanisms, including **Self-Attention** and **Multi-Head Attention**.

The goal of this project is to understand how attention works in deep learning models, especially in transformer-based architectures.

---

## Project Structure

```text
attention-mechanism/
│
├── self_attention.py
├── multi_head_attention.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Self-Attention

The `self_attention.py` file demonstrates the basic idea of self-attention.

It creates three matrices:

- `Q` — Query
- `K` — Key
- `V` — Value

Then it computes attention scores using:

```python
attention_scores = Q @ K.T
```

The attention scores are normalized using softmax:

```python
attention_weights = F.softmax(attention_scores, dim=-1)
```

Finally, the output is computed by multiplying the attention weights with the value matrix:

```python
output = attention_weights @ V
```

---

## Multi-Head Attention

The `multi_head_attention.py` file implements a simple version of multi-head attention using PyTorch.

Multi-head attention allows the model to focus on different parts of the input sequence at the same time.

The implementation includes:

- Linear layers for Query, Key, and Value
- Splitting embeddings into multiple heads
- Scaled dot-product attention
- Optional masking
- Combining the heads back into one output

---

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The main dependency is:

```text
torch
```

---

## How to Run

To run the self-attention example:

```bash
python self_attention.py
```

To run the multi-head attention example:

```bash
python multi_head_attention.py
```

---

## Example Output

The scripts print attention weights, attention scores, and output tensors.

Example:

```text
Attention Weights:
tensor(...)

Output:
tensor(...)
```

---

## Notes

This project is for learning purposes. It is a simple implementation meant to help understand the core ideas behind attention mechanisms.

In real transformer models, attention is usually combined with other components such as:

- Positional encoding
- Feed-forward neural networks
- Layer normalization
- Residual connections
- Dropout
