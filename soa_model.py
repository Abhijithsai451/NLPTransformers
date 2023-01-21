import logging
import warnings

import numpy as np
import pandas as pd
import regex as re
import tensorflow as tf
import tqdm
from keras import Input, Model
from keras.optimizers import SGD
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoTokenizer, TFAutoModel, TFRobertaForSequenceClassification

import config_file

# Neural Machine Translation #

# Using the pretrained model like BERT or State of Art Model.
# Using the Model "XLM-roberta-base-language-detection"
# About Model: This model can detect the language. It can identify many languages likes spanish, German, French, italian
# Russian, Turkish, Portuguese which are exactly what we are training the model with.
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='[NLP Transformer] %(process)d-%(levelname)s-%(message)s')
logging.info("TensorFlow Version:  " + tf.__version__)
logging.info("GPU Availability:  " + tf.test.gpu_device_name())
AUTO = tf.data.experimental.AUTOTUNE

# Data Preparation
logging.info("Reading the CONFIG file [config_file.py]")

logging.info("Importing the Data from Dataset [Jigsaw Toxic Comments]")
data_path = config_file.data_path
es_data = pd.read_csv(data_path + "/jigsaw-toxic-comment-train-google-es-cleaned.csv")
fr_data = pd.read_csv(data_path + "/jigsaw-toxic-comment-train-google-fr-cleaned.csv")
it_data = pd.read_csv(data_path + "/jigsaw-toxic-comment-train-google-it-cleaned.csv")
pt_data = pd.read_csv(data_path + "/jigsaw-toxic-comment-train-google-pt-cleaned.csv")
ru_data = pd.read_csv(data_path + "/jigsaw-toxic-comment-train-google-ru-cleaned.csv")
tr_data = pd.read_csv(data_path + "/jigsaw-toxic-comment-train-google-tr-cleaned.csv")
logging.info("Successfully imported the data from the [Jigsaw Toxic Comments] Dataset")

es_data['language'] = 'spanish'
fr_data['language'] = 'french'
it_data['language'] = 'italian'
pt_data['language'] = 'portuguese'
ru_data['language'] = 'russian'
tr_data['language'] = 'turkish'
logging.info("Created the individual dataframes for comments on each language")

corpus = []


def clean_data(data):
    for i in range(len(data)):
        text_data = re.sub('[^a-zA-Z]', ' ', data['comment_text'][i])
        text_data = text_data.lower()
        text_data = text_data.split()
        ps = PorterStemmer()
        text_data = [ps.stem(word) for word in text_data if not word in set(stopwords.words(data['language'][i]))]
        data['clean_text'][i] = text_data
        text_data = ' '.join(text_data)
        corpus.append(text_data)
        data['clean_text'][i] = text_data
    return corpus, data


def preprocess_dataframe(data):
    data = data[:100]
    data['clean_text'] = ""
    data = data[['comment_text', 'language', 'clean_text']]
    corpus, data = clean_data(data)
    return data


def encode_labels(dataframe):
    encoder = OrdinalEncoder()
    dataframe['language'] = encoder.fit_transform(dataframe[['language']])
    dataframe['language'] = dataframe['language']/6
    return dataframe

logging.info("Preprocessing the individual dataframes -> Removing Special Chars, Stemming, Removing Stop words")
esp_data = preprocess_dataframe(es_data)
frh_data = preprocess_dataframe(fr_data)
itl_data = preprocess_dataframe(it_data)
ptg_data = preprocess_dataframe(pt_data)
rus_data = preprocess_dataframe(ru_data)
trh_data = preprocess_dataframe(tr_data)
logging.info("Concatinating the individual dataframes to a single main dataframe.")
data = pd.concat([
    esp_data[['clean_text', 'language']],
    frh_data[['clean_text', 'language']],
    itl_data[['clean_text', 'language']],
    ptg_data[['clean_text', 'language']],
    rus_data[['clean_text', 'language']],
    trh_data[['clean_text', 'language']]
], ignore_index=True)

data_final = encode_labels(data)


# Data Encoding with Tokenizer

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


def encode_data(texts, tokenizer, maxlen=150):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_mask=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        truncation=True,
        max_length=maxlen
    )

    return np.array(enc_di['input_ids'])

logging.info("Extracting the 'ivanlau/language-detection-fine-tuned-on-xlm-roberta-base' tokenizer")
tokenizer = AutoTokenizer.from_pretrained("ivanlau/language-detection-fine-tuned-on-xlm-roberta-base")

# Train Test Validation Split data
logging.info("train test split the data ")
X_train, X_test, y_train, y_test = train_test_split(data_final['clean_text'], data['language'], test_size=0.25,
                                                    random_state=42)

X_train = encode_data(X_train.tolist(), tokenizer, maxlen=config_file.sent_leng)
X_test = encode_data(X_test.tolist(), tokenizer, maxlen=config_file.sent_leng)


# Creating Tensorflow Dataset
logging.info("Creating the tensorflow datasets: train_data")
train_data = (tf.data.Dataset
              .from_tensor_slices((X_train, y_train))
              .repeat()
              .shuffle(600)
              .batch(config_file.BATCH_SIZE)
              .prefetch(AUTO)
              )
logging.info("Creating the tensorflow datasets: test_data")
test_data = (tf.data.Dataset
             .from_tensor_slices((X_test, y_test))
             .batch(config_file.BATCH_SIZE)
             )


def build_model( max_len = config_file.sent_leng):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    transformer = TFRobertaForSequenceClassification.from_pretrained("xlm-mlm-en-2048")
 
    output_sequence = transformer(input_ids).logits

    model = Model(inputs = input_ids, outputs = output_sequence)
    model.compile(SGD(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

logging.info("building the 'ivanlau/language-detection-fine-tuned-on-xlm-roberta-base: Pretrained'")
model = build_model()
logging.info("Created the 'XLM Roberta language detection model: Pretrained'")
print(model.summary())

# Training
n_steps = X_train.shape[0] // config_file.BATCH_SIZE
print("n_steps -->>>>>", n_steps)
result = model.fit(train_data, steps_per_epoch=n_steps, validation_data=test_data, epochs=config_file.EPOCHS)

# Required Models to Research
# RoBERT
# GottBERT
# mBERT
# XLM-R