# Seq2SQL SQLNet Optimized

Welcome to the optimized repository for [Seq2SQL](https://arxiv.org/abs/1709.00103) and [SQLNet](https://arxiv.org/abs/1711.04436). This version has been updated to be compatible with Python 3.11.3. The original implementation, written in Python 2.7, can be found [here](https://github.com/xiaojunxu/SQLNet).

## Installation

Before you begin, you'll need to download and extract the GloVe word embeddings. This can be done by running download_embedding.py:

```bash
python download_embedding.py
```
## Alter questions

To generate altered questions through GPT-4 API, run alter_questions.py.

## Training

To start the training process, you will need to run the train.py script. You can modify the configuration parameters within this script to suit your needs.

## Testing

Once you have a trained model, you can test it using the test.py script. Like with train.py, you are free to adjust the configuration parameters in this script.
