import logging
import warnings

import pandas as pd
import regex as re
import tensorflow as tf
from nltk import PorterStemmer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import config_file
import tqdm
import numpy as np
from  sklearn.model_selection import train_test_split


# Using the pretrained model like BERT or State of Art Model.
# Using the Model "XLM-roberta-base-language-detection"
# About Model: This model can detect the language. It can identify many languages likes spanish, German, French, italian
# Russian, Turkish, Portuguese which are exactly what we are training the model with.
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='[NLP Transformer] %(process)d-%(levelname)s-%(message)s')
logging.info("TensorFlow Version:  " + tf.__version__)
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


esp_data = preprocess_dataframe(es_data)
frh_data = preprocess_dataframe(fr_data)
itl_data = preprocess_dataframe(it_data)
ptg_data = preprocess_dataframe(pt_data)
rus_data = preprocess_dataframe(ru_data)
trh_data = preprocess_dataframe(tr_data)

data = pd.concat([
    esp_data[['comment_text', 'language', 'clean_text']],
    frh_data[['comment_text', 'language', 'clean_text']],
    itl_data[['comment_text', 'language', 'clean_text']],
    ptg_data[['comment_text', 'language', 'clean_text']],
    rus_data[['comment_text', 'language', 'clean_text']],
    trh_data[['comment_text', 'language', 'clean_text']]
], ignore_index=True)


# Data Encoding with Tokenizer

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)

def encode_data(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )

    return np.array(enc_di['input_ids'])

tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

# Train Test Validation Split data
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'],data['language'], test_size=0.33, random_state=42)




X_train = encode_data(X_train.clean_text.values, tokenizer,maxlen= config_file.sent_leng)
X_test = encode_data(X_test,tokenizer,maxlen=config_file.sent_leng)

y_test =




