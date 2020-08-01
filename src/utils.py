import numpy as np

import re
import os

from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional, Input
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config


def get_model(embeddings):
    model = Sequential()
    model.add(Embedding(config.VOCAB_SIZE + 1, 100,
                        embeddings_initializer=embeddings,
                        input_length=config.MAX_LEN,
                        trainable=False))
    model.add(Bidirectional(LSTM(config.MAX_LEN)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def get_model(embeddings):
    text = Input(shape=(config.MAX_LEN,))
    embed = Embedding(config.VOCAB_SIZE + 1, 100,
                      embeddings_initializer=embeddings,
                      input_length=config.MAX_LEN,
                      trainable=False)(text)
    x = Bidirectional(LSTM(config.MAX_LEN))(embed)
    x = Dropout(0.2)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(text, preds)
    return model

def NEAModel(embeddings, initial_mean_value):
    model = Sequential()
    model.add(Embedding(config.VOCAB_SIZE, 100,
                        embeddings_initializer=embeddings,
                        input_length=config.MAX_LEN,
                        trainable=False))
    model.add(LSTM(config.MAX_LEN, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(MeanOverTime(mask_zero=True))
    model.add(Dense(1))
    # bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
    # model.layers[-1].b.set_value(bias_value)
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def get_model_prompt():
    text = Input(shape=(config.MAX_LEN,))
    embed_1 = Embedding(config.VOCAB_SIZE + 1, 100,
                        input_length=config.MAX_LEN,
                        trainable=False)(text)
    prompt = Input(shape=(config.MAX_LEN_PROMPT,))
    embed_2 = Embedding(config.VOCAB_SIZE + 1, 100,
                        input_length=config.MAX_LEN_PROMPT,
                        trainable=False)(prompt)
    conc = Concatenate(axis=-1)([embed_1, embed_2])
    lstm = Bidirectional(LSTM(config.MAX_LEN))(conc)
    drop = Dropout(0.4)(lstm)
    dense = Dense(1)(drop)
    output = Activation('sigmoid')(dense)
    model = Model([prompt, text], output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def preprocess(text_raw, tokenizer, MAX_LEN):
    text_encoded = tokenizer.texts_to_sequences(text_raw)

    text_array = pad_sequences(
        text_encoded,
        maxlen=MAX_LEN,
        padding='post')

    return text_array


def load_embedding_matrix(tokenizer, glove_path):
    embeddings_index = {}
    if glove_path == '../input/glove.6B.100d.txt':
        dim = 100
    if glove_path == '../input/glove.840B.300d.txt':
        dim = 300
    # glove_path = os.path.join('../input', glove_path)

    print("Loading embedding matrix...")

    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((config.VOCAB_SIZE + 1, dim))

    word_index = tokenizer.word_index
    for word, i in word_index.items():
        if i > config.VOCAB_SIZE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Embedding matrix loaded!")
    print('G Word embeddings:', len(embeddings_index))
    print(embedding_matrix.shape)


def replace_label(e):
    p_caps = re.compile(r'@caps\d{0,2}', re.I)
    p_city = re.compile(r'@city\d{0,2}', re.I)
    p_date = re.compile(r'@date\d{0,2}', re.I)
    p_dr = re.compile(r'@dr\d{0,2}', re.I)
    p_location = re.compile(r'@location\d{0,2}', re.I)
    p_money = re.compile(r'@money\d{0,2}', re.I)
    p_month = re.compile(r'@month\d{0,2}', re.I)
    p_num = re.compile(r'@num\d{0,2}', re.I)
    p_organization = re.compile(r'@organization\d{0,2}', re.I)
    p_percent = re.compile(r'@percent\d{0,2}', re.I)
    p_person = re.compile(r'@person\d{0,2}', re.I)
    p_state = re.compile(r'@state\d{0,2}', re.I)
    p_time = re.compile(r'@time\d{0,2}', re.I)
    p_email = re.compile(r'@email\d{0,2}', re.I)

    text = e
    text = p_caps.sub('LABEL_CAPS', text)
    text = p_city.sub('LABEL_CITY', text)
    text = p_date.sub('LABEL_DATE', text)

    text = p_dr.sub('LABEL_DR', text)
    text = p_location.sub('LABEL_LOCATION', text)
    text = p_money.sub('LABEL_MONEY', text)
    text = p_month.sub('LABEL_MONTH', text)
    text = p_num.sub('LABEL_NUM', text)
    text = p_organization.sub('LABEL_ORGANIZATION', text)
    text = p_percent.sub('LABEL_PERCENT', text)
    text = p_person.sub('LABEL_PERSON', text)
    text = p_state.sub('LABEL_STATE', text)
    text = p_time.sub('LABEL_TIME', text)
    text = p_email.sub('LABEL_EMAIL', text)
    return text
