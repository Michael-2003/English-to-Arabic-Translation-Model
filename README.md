# English-to-Arabic Translation Model

This project implements a sequence-to-sequence neural network model with attention for translating English sentences to Arabic. The model is built using TensorFlow and Keras, leveraging LSTM layers and a custom attention mechanism for improved translation quality.

## Project Overview

The model is trained on a dataset of English-Arabic sentence pairs and utilizes an encoder-decoder architecture with an attention mechanism. The encoder processes the input sentence, and the decoder generates the translated sentence, focusing on relevant parts of the input sentence through the attention layer.

### Key Features
- **Sequence-to-sequence architecture** with an encoder and decoder
- **Bidirectional LSTM** in the encoder for better context understanding
- **Attention Mechanism** to improve translation by focusing on relevant input segments
- **Tokenization and Padding** for both input (English) and target (Arabic) sentences

## Setup and Installation

### Prerequisites

- Python 3.x
- TensorFlow and Keras
- Numpy and Pandas

To install required libraries:
```bash
pip install tensorflow pandas numpy
```


### Dataset

Download the dataset from Kaggle and save it in the specified path. In this example, the dataset is stored as `ara_eng.txt` with columns `English` and `Arabic`.

[Kaggle Dataset: Arabic-to-English Translation Sentences](https://www.kaggle.com/datasets/samirmoustafa/arabic-to-english-translation-sentences)


## Model Architecture

1. **Encoder**: Processes the English sentence with a Bidirectional LSTM, capturing context in both directions.
2. **Decoder**: Generates Arabic words sequentially, utilizing previous states and attention-weighted encoder outputs.
3. **Attention Layer**: Enhances translation accuracy by emphasizing relevant words in the input sequence.
4. **Training Configuration**: Uses `sparse_categorical_crossentropy` as the loss function and the `Adam` optimizer.

## Files

- **english_translate_to_arabic.ipynb**: Contains all code for loading the data, building, training, and testing the model.
- **ara_eng.txt**: Text file containing English-Arabic sentence pairs for training.
