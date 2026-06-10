# miniGPT

A decoder-only transformer implemented from scratch in PyTorch, designed for character-level language modeling. This project specifically focuses on generating "babbling" Shakespeare-like text by training on the Tiny Shakespeare dataset.

## Overview

This repository features a GPT-style model implementation that predicts the next character in a sequence based on the preceding context. It uses a character-level bigram model foundation scaled up into a full transformer architecture, utilizing a context length of 256 characters to predict the 257th character.

## Architecture Details

- **Parameters:** ~10.7 Million (`10.78M`)
- **Context Length (Block Size):** 256 tokens (characters)
- **Embedding Dimension:** 384
- **Attention Heads:** 6 heads (each of size 64)
- **Transformer Layers:** 6 layers
- **Dropout Rate:** 0.2
- **Optimizer:** AdamW (Learning rate: `3e-4`)

## Features

- **Character-level Tokenizer:** A simple but effective character-to-integer mapping built directly from the unique characters in the dataset.
- **Multi-Head Self-Attention:** Core transformer attention mechanisms built entirely from scratch using PyTorch primitives.
- **Feed-Forward Networks:** Included inside each transformer block to add non-linearity.
- **Layer Normalization & Residual Connections:** Stabilizes and speeds up deep network training.
- **Text Generation:** Built-in method to auto-regressively sample new text from the trained model.

## Dataset

The model trains on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). The script expects this file to be named `input.txt` and placed in the project's root directory or the directory where the script is run.

## Compare Outputs
I've added all the outputs in `output/` folder after each architectural changes to compare the performance of the model with progressive architecture.

## Setup & Usage

### Prerequisites
- Python 3.x
- PyTorch (`torch`)

### Running the Model

1. Make sure you have the `input.txt` file in your working directory. If you don't have it, you can download it using:
   ```bash
   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   ```
2. Run the main training script:
   ```bash
   python gpt.py
   ```

The script will train the model for 5000 iterations, printing the training and validation loss every 500 steps. Once training is complete, it will generate and print a 500-character sample of text.
