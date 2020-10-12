# LSTM Based Essay Grading
Using LSTMs to automatically grade [Kaggle's ASAP Essays](https://www.kaggle.com/c/asap-aes).

## Models
Pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1EhMNLRuOnNOSHutVZrHMvkSiMAtlOXJ7?usp=sharing).

## Training
1. Download GloVe's 6B.100d word vectors from [here](https://nlp.stanford.edu/projects/glove/). You can also train on 840B.300d for better results.
2. Set your desired config in config.py and simply run the training using train.py.
3. You can serve you own trained models by modifying models.config inside the /models directory.

## Usage

In 1st Terminal

```
git clone https://github.com/tanmaypandey7/essay-grading
cd essay-grading
pip install -r requirements.txt
python app.py
```

In 2nd Terminal
```
sudo service docker start
# For GPU
docker run -t --rm --gpus all --name=tf-serving -p  8501:8501     \
-v "$(pwd)/models:/models/" tensorflow/serving:latest-gpu   \
     --model_config_file=/models/models.config
# For CPU
docker run -p  8501:8501 \
-v "$(pwd)/models:/models/" tensorflow/serving:latest   \
     --model_config_file=/models/models.config
```