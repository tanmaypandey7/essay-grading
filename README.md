# LSTM Based Essay Grading
Using LSTMs to automatically grade [Kaggle's ASAP Essays](https://www.kaggle.com/c/asap-aes).

## Models
Pre-trained models can be downloaded from [here](https://www.kaggle.com/c/asap-aes).

## Training
1. Download GloVe's 6B.100d word vectors from [here](https://nlp.stanford.edu/projects/glove/). You can also train on 840B.300d for better results.
2. Set your desired config in config.py and simply run the training using train.py.
3. You can serve you own trained models by modifying models.config inside the /models directory.
