# Transformer: Attention Is All You Need

A PyTorch implementation of the Transformer architecture from the seminal paper ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). This implementation focuses on English-to-German machine translation using the Multi30k dataset.

## Overview

This project replicates the Transformer architecture introduced in the "Attention Is All You Need" paper, which revolutionized sequence-to-sequence modeling by relying entirely on attention mechanisms without recurrent or convolutional layers. The model uses multi-head self-attention, positional encoding, and feedforward networks to achieve state-of-the-art translation performance.

## Dataset

This implementation uses the **Multi30k** dataset, a multilingual extension of the Flickr30k dataset containing English-German sentence pairs. The dataset is available on Hugging Face:

- **Dataset**: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)

The dataset is automatically downloaded when you run the training script.

## Project Structure

### Core Components

- **`transformer.py`** - Main Transformer model class that combines encoder and decoder stacks
- **`encoder.py`** - Encoder stack implementation with multiple encoder layers
- **`decoder.py`** - Decoder stack implementation with multiple decoder layers
- **`encoder_layer.py`** - Single encoder layer with multi-head attention and feedforward network
- **`decoder_layer.py`** - Single decoder layer with masked multi-head attention, encoder-decoder attention, and feedforward network
- **`attention.py`** - Multi-head attention mechanism implementation
- **`feedforward.py`** - Position-wise feedforward network
- **`positional_encoding.py`** - Sinusoidal positional encoding for sequence position information

### Data & Training

- **`data_loader.py`** - Dataset loading, tokenization, vocabulary building, and batch creation utilities
- **`train.py`** - Training script with learning rate scheduling, gradient clipping, and checkpoint saving
- **`translate.py`** - Inference script for translating English sentences to German

### Configuration

- **`requirements.txt`** - Python dependencies including PyTorch, datasets, and spaCy models

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd "Attention is all you need"
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy language models (required for tokenization):
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Usage

### Training

Train the Transformer model on the Multi30k dataset:

```bash
python train.py
```

The training script will:
- Automatically download and preprocess the Multi30k dataset
- Build vocabularies from the training data
- Train the model with learning rate warmup and gradient clipping
- Save the trained model to `transformer_safe.pth`

**Training Configuration** (can be modified in `train.py`):
- `D_MODEL`: 256 (model dimension)
- `NUM_LAYERS`: 2 (number of encoder/decoder layers)
- `NUM_HEADS`: 4 (number of attention heads)
- `D_FF`: 256 (feedforward dimension)
- `BATCH_SIZE`: 128
- `EPOCHS`: 10
- `LR`: 1e-4 (learning rate)

### Translation (Inference)

Translate English sentences to German using the trained model:

```bash
python translate.py "Your English sentence here"
```

Or run interactively:
```bash
python translate.py
```

Then type English sentences and press Enter to get German translations.

## Model Architecture

The Transformer follows the architecture described in the paper:

- **Encoder**: Stack of identical layers, each containing:
  - Multi-head self-attention mechanism
  - Position-wise feedforward network
  - Residual connections and layer normalization

- **Decoder**: Stack of identical layers, each containing:
  - Masked multi-head self-attention
  - Multi-head encoder-decoder attention
  - Position-wise feedforward network
  - Residual connections and layer normalization

- **Positional Encoding**: Sinusoidal embeddings to provide sequence order information

## Key Features

- Multi-head attention mechanism
- Positional encoding
- Residual connections and layer normalization
- Learning rate warmup scheduling
- Gradient clipping for training stability
- Label smoothing
- Support for CUDA, MPS (Apple Silicon), and CPU

## Model Parameters

The model contains approximately **7.6 million trainable parameters**, making it suitable for training on consumer hardware.

### Training Performance

- **Hardware**: Mac M4 Pro (Apple Silicon)
- **Dataset Size**: ~30,000 training samples (Multi30k)
- **Training Time**: ~5 minutes for 10 epochs
- **Device**: MPS (Metal Performance Shaders) acceleration