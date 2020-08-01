import pandas as pd

import os

import config
import utils

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime


def run():
    df = pd.read_csv(config.INPUT_FILE)

    if config.TRAIN_PROMPT:
        df = df[['prompt', 'essay', config.TRAIN_FOR]]
    else:
        df = df[['essay', config.TRAIN_FOR]]

    df['essay_cleaned'] = df['essay'].apply(utils.replace_label)

    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE)
    if config.TRAIN_PROMPT:
        tokenizer.fit_on_texts(df['prompt'])
    tokenizer.fit_on_texts(df['essay_cleaned'])

    X = utils.preprocess(df['essay_cleaned'], tokenizer, config.MAX_LEN)
    if config.TRAIN_PROMPT:
        X_prompt = utils.preprocess(df['prompt'],
                                    tokenizer,
                                    config.MAX_LEN_PROMPT)

    y = df[config.TRAIN_FOR].values

    # Uncomment if getting "DNN implementation Not Found" Error
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    embeddings = utils.load_embedding_matrix(tokenizer, config.GLOVE_PATH)

    if config.TRAIN_PROMPT:
        model = utils.get_model_prompt()
    else:
        model = utils.get_model(embeddings)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    mcp_save = ModelCheckpoint(
        filepath=f'../models/model-PROMPT_{config.TRAIN_PROMPT}_{config.TRAIN_FOR}_epochs_{config.EPOCHS}_{datetime.now()}.h5',
        save_best_only=True,
        monitor='val_mae',
        mode='min',
        verbose=1)

    earlyStopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min')

    if config.TRAIN_PROMPT:
        history = model.fit([X_prompt, X],
                            y,
                            batch_size=config.BATCH_SIZE,
                            epochs=config.EPOCHS,
                            validation_split=.2,
                            verbose=1,
                            callbacks=[mcp_save, earlyStopping])
    else:
        history = model.fit(X,
                            y,
                            batch_size=config.BATCH_SIZE,
                            epochs=config.EPOCHS,
                            validation_split=.3,
                            verbose=1,
                            shuffle=True,
                            callbacks=[mcp_save, earlyStopping])
    # print(model.summary())

    '''
    For saving pickle model
    with open(f'../models/model-TRAIN_PROMPT-{config.TRAIN_PROMPT}-\
    {config.TRAIN_FOR}-epochs-{config.EPOCHS}-\
    {datetime.now()}.pickle', 'wb') as handle:
        pickle.dump(history.history, handle)

    with open(f'../models/tokenizer_essays.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    '''

    # Saving the model
    if config.TRAIN_PROMPT:
        MODEL_DIR = f"../models/prompt-essay/PROMPT_{config.TRAIN_FOR}"
    else:
        MODEL_DIR = f"../models/{config.TRAIN_FOR}"
    version = "1"
    export_path = os.path.join(MODEL_DIR, version)
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
                            model,
                            export_path,
                            overwrite=True,
                            include_optimizer=True,
                            save_format=None,
                            signatures=None,
                            options=None
                            )


if __name__ == '__main__':
    run()
