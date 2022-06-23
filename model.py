import configparser
import logging
import time
import warnings

import nltk
import numpy as np
import pandas as pd
import regex as re
import tensorflow as tf
from keras.layers import Dense, Embedding, BatchNormalization, Flatten
from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.data import Dataset

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='[NLP Transformer] %(process)d-%(levelname)s-%(message)s')
logging.info("TensorFlow Version:  " + tf.__version__)

config = configparser.RawConfigParser()
logging.info("Reading the CONFIG file [config_file.properties]")
config.read('config_file.properties')

logging.info("Importing the Data from Dataset [Jigsaw Toxic Comments]")
mul_path = config.get('run_properties', 'data_path')
es_data = pd.read_csv(mul_path + "/jigsaw-toxic-comment-train-google-es-cleaned.csv")
fr_data = pd.read_csv(mul_path + "/jigsaw-toxic-comment-train-google-fr-cleaned.csv")
it_data = pd.read_csv(mul_path + "/jigsaw-toxic-comment-train-google-it-cleaned.csv")
pt_data = pd.read_csv(mul_path + "/jigsaw-toxic-comment-train-google-pt-cleaned.csv")
ru_data = pd.read_csv(mul_path + "/jigsaw-toxic-comment-train-google-ru-cleaned.csv")
tr_data = pd.read_csv(mul_path + "/jigsaw-toxic-comment-train-google-tr-cleaned.csv")
logging.info("Successfully imported the data from the [Jigsaw Toxic Comments] Dataset")

vocab_size = int(config.get('run_properties', 'vocab_size'))
sent_leng = int(config.get('run_properties', 'sent_leng'))
embedding_vector_features = config.get('run_properties', 'embedding_vector_features')
num_lang = int(config.get('run_properties', 'num_lang'))
EPOCHS = int(config.get('run_properties', 'EPOCHS'))
BATCH_SIZE = int(config.get('run_properties', 'BATCH_SIZE'))
MODEL_SIZE = int(config.get('run_properties', 'MODEL_SIZE'))
num_layers = int(config.get('run_properties','num_layers'))
h = int(config.get('run_properties','h'))
logging.info("Imported the properties from the config file ")

es_data['language'] = 'spanish'
fr_data['language'] = 'french'
it_data['language'] = 'italian'
pt_data['language'] = 'portuguese'
ru_data['language'] = 'russian'
tr_data['language'] = 'turkish'

# ----------------------------------------------------------------------------------------------------------------------
## Data Cleansing and PreProcessing
#   1.  Cleansing the Data i.e removing special chars
#   2.  Stemming the data
#   3.  Removing stop words
#   4.  Encoding and Embedding

#* We labelled the data with its respective language i.e spanish, french, italian etc
#* Using the labels we created, we import the stop words from that respective language and clean the data.
#* After cleaning all the dataframes, I concatinated all six dataframes into a single for further processing
# ----------------------------------------------------------------------------------------------------------------------

corpus = []
nltk.download('stopwords')


def clean_data(data):
    logging.info("cleaning the data in the dataframe ")
    logging.info("checking for special chars,stop words, convert to lower case")
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
    logging.info("Preprocessing the data frame ")
    data['clean_text'] = ""
    data = data[['comment_text', 'language', 'clean_text']]
    corpus, data = clean_data(data)
    logging.info("cleaned the dataframe ")
    return data


esp_data = preprocess_dataframe(es_data)
frh_data = preprocess_dataframe(fr_data)
itl_data = preprocess_dataframe(it_data)
ptg_data = preprocess_dataframe(pt_data)
rus_data = preprocess_dataframe(ru_data)
trh_data = preprocess_dataframe(tr_data)

logging.info("Successfully cleaned the data and Concatenating all the dataframes")
clean_data = pd.concat([
    esp_data[['comment_text', 'language', 'clean_text']],
    frh_data[['comment_text', 'language', 'clean_text']],
    itl_data[['comment_text', 'language', 'clean_text']],
    ptg_data[['comment_text', 'language', 'clean_text']],
    rus_data[['comment_text', 'language', 'clean_text']],
    trh_data[['comment_text', 'language', 'clean_text']]], ignore_index=True)
logging.info("Concatenated all the dataframes")
print(clean_data[1:10])

# Data Preparation
logging.info("Preparing the data for further processing with the Transformers")
transformer_data = pd.DataFrame()
transformer_data = clean_data[['clean_text', 'language']]
logging.info("Appending the tag [<start>] to the transformer input for encoding purpose")
transformer_data['clean_text'] = '<start> ' + transformer_data['clean_text']
logging.info("Sample data in the final dataframe is -->")
transformer_data.head()

# Tokenizing the data
sentences = []
labels = []
def tokenize_data(sentences):
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')
    return padded_sequences
def encode_transformer_data(data):
    logging.info("Encoding the target labels")
    labelEncoder = LabelEncoder()
    data['label'] = labelEncoder.fit_transform(data['language'])
    return data['label']

logging.info("Extracting the sentences from the data")
sentences = transformer_data['clean_text']
padded_sequences = tokenize_data(sentences)
labels = encode_transformer_data(transformer_data)
print(labels[0].shape)
print(padded_sequences.shape)

data = Dataset.from_tensor_slices((padded_sequences, labels))
data = data.shuffle(20).batch(BATCH_SIZE)
'''for d in data:
    print(d)
    break'''
# ----------------------------------------------------------------------------------------------------------------------
# Creating Transformer Model
#   1.Positional Encoding
#   2. Multi Head Attention Layer
#   3. Encoder
#   4. Decoder
# ----------------------------------------------------------------------------------------------------------------------
logging.info("creating the Transfomers model")
def positional_encoding(pos, d_model):
    PE = np.zeros((1,d_model))
    for i in range(d_model):
        if i%2 ==0:
            PE[:,i] = np.sin(pos/10000 **(i/d_model))
        else:
            PE[:,i] = np.cos(pos/10000 **((i-1)/d_model))
    return PE
max_length = len(clean_data)
pes = []
for i in range(max_length):
    pes.append(positional_encoding(i,MODEL_SIZE))
pes = np.concatenate(pes, axis=0)
logging.info("creating the positional Encoding")
pes = tf.constant(pes,dtype=tf.float32)
#print(pes[:10, :1])

logging.info("Creating the Multi Head Attention Layer class")
class MultiHeadAttention(tf.keras.Model):
    def __init__(self, MODEL_SIZE, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = MODEL_SIZE // h
        self.key_size = MODEL_SIZE // h
        self.value_size = MODEL_SIZE // h

        self.h = h

        self.wq = [Dense(self.query_size) for _ in range(h)]
        self.wk = [Dense(self.key_size) for _ in range(h)]
        self.wv = [Dense(self.value_size) for _ in range(h)]
        self.wo = Dense(MODEL_SIZE)

    def call(self, query, value):
        heads = []

        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))

            alignment = tf.nn.softmax(score, axis=2)

            head = tf.matmul(alignment, self.wv[i](value))

            heads.append(head)
            heads = tf.concat(heads, axis=2)
            heads = self.wo(heads)

            return heads

# ----------------------------------------------------------------------------------------------------------------------
# Encoder and Decoder Implementation
#   We have both the positional encoding and MultiHeadAttention blocks of the transformer
#   We need to use the above including the feed forward and Normalization blocks to implement the Encoder Architecture
#-----------------------------------------------------------------------------------------------------------------------

class Encoder(tf.keras.Model):
    logging.info("Creating the Encoder class")
    def __init__(self, vocab_size, MODEL_SIZE, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = MODEL_SIZE
        self.num_layers = num_layers
        self.h = h

        # Embedding layer
        self.embedding = Embedding(vocab_size, MODEL_SIZE)

        # Number of Attention Layers and Normalization Layers
        self.attention = [MultiHeadAttention(MODEL_SIZE, h) for _ in range(num_layers)]
        self.att_norm = [BatchNormalization() for _ in range(num_layers)]

        # Number of Feed Forward Networks and Normalization Layers
        self.dense1 = [Dense(MODEL_SIZE * 4, activation='relu') for _ in range(num_layers)]
        self.dense2 = [Dense(MODEL_SIZE) for _ in range(num_layers)]
        self.ffn_norm = [BatchNormalization() for _ in range(num_layers)]

    # Forward Path
    def call(self, sequence):
        sub_in = []

        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            sub_in.append(embed + pes[i, :1])
        sub_in = tf.concat(sub_in, axis=1)

        for i in range(self.num_layers):
            sub_out = []

            for j in range(sub_in.shape[1]):
                attention = self.attention[i](tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)
                sub_out.append(attention)
            sub_out = tf.concat(sub_out, axis=1)

            sub_out = sub_in + sub_out
            sub_out = self.att_norm[i](sub_out)

            ffn_in = sub_out
            ffn_out = self.dense2[i](self.dense1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out
        logging.info("Created the Encoder class")
        return ffn_out


class Decoder(tf.keras.Model):
    logging.info("Creating the Decoder class")
    def __init__(self, vocab_size, MODEL_SIZE, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = MODEL_SIZE
        self.num_layers = num_layers
        self.h = h

        self.embedding = Embedding(vocab_size, MODEL_SIZE)
        self.att_bot = [MultiHeadAttention(MODEL_SIZE, h) for _ in range(num_layers)]
        self.attb_norm = [BatchNormalization() for _ in range(num_layers)]
        self.att_mid = [MultiHeadAttention(MODEL_SIZE, h) for _ in range(num_layers)]
        self.att_mid_norm = [BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [Dense(MODEL_SIZE * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [Dense(MODEL_SIZE) for _ in range(num_layers)]
        self.ffn_norm = [BatchNormalization() for _ in range(num_layers)]
        self.flatten_layer = Flatten()
        self.dense = Dense(vocab_size)


    def call(self, sequence, encoder_output):
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = []
        print("Sequence shape is ----->>>>>>",sequence.shape)
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            embed_out.append(embed + pes[i, :])

        embed_out = tf.concat(embed_out, axis=1)

        bot_sub_in = embed_out

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            bot_sub_out = []
            for j in range(bot_sub_in.shape[1]):
                values = bot_sub_in[:, :j]
                attention = self.att_bot[i](tf.expand_dims(bot_sub_in[:, j]), values)

                bot_sub_out.append(attention)
            bot_sub_out = tf.concat(bot_sub_out, axis=1)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attb_norm[i](bot_sub_out)
            logging.info("Created the positional encoding and Bottom Mulit head Layer ")

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out
            mid_sub_out = []
            for j in range(mid_sub_in.shape[1]):
                attention = self.att_mid[i](
                    tf.expand_dims(mid_sub_in[:, j]), encoder_output)
                mid_sub_out.append(attention)

            mid_sub_out = tf.concat(mid_sub_out, axis=1)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.att_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out
            flat_out = self.flatten_layer(ffn_out)
        logits = self.dense(flat_out)
        print(logits.shape)
        logging.info("Created the Decoder class")
        return logits

# Encoder Model
enc_model = Encoder(vocab_size, MODEL_SIZE, num_layers , h)

# Decoder Model
dec_model = Decoder(1, MODEL_SIZE, num_layers, h)


# ----------------------------------------------------------------------------------------------------------------------
# Model Training
#   Defining Loss Function, Metrics
#   Train Step
#   Prediction Step
#   Training the model -> invoking train step methods
#-----------------------------------------------------------------------------------------------------------------------

# Loss Function
crossentropy = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False)

def loss_function(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(source_seq, target_seq, enc_model, dec_model):
    with tf.GradientTape() as tape:
        encoder_output = enc_model(source_seq)
        decoder_output = dec_model(target_seq, encoder_output)
        print(decoder_output.shape)
        print(target_seq.shape)
        loss = loss_function(target_seq, decoder_output)

    variables = enc_model.trainable_variables + dec_model.trainable_variables
    gradients = tape.gradients(loss, variables)
    optimizer.apply_gradient(zip(gradients, variables))
    return loss


def predict(test_seq):
    if test_seq == None:
        test_seq = transformer_data['clean_text'][np.random.choice(len(transformer_data))]
    print("Predicting the language of the sentence ", test_seq)
    test_seq = tokenize_data(test_seq)
    print("test_seq after tokenizing and padding ", test_seq)
    encoder_output = enc_model(test_seq)
    new_seq = tf.zeros((6,))
    dec_output = dec_model(new_seq, encoder_output)

    return dec_output


start_time = time.time()

for e in range(EPOCHS):
    for batch, (source_seq, target_seq) in enumerate(data.take(-1)):

        loss = train_step(source_seq, target_seq, enc_model, dec_model)

    print('Epoch {} Loss {:.4f}'.format(e+1, loss.numpy()))

    if (e+1) % 10 == 0:
        end_time = time.time()
        print('Elapsed Time: {:.2f}s'.format((end_time - start_time)/(e+1)))
        try:
            predict()
        except Exception as e:
            print(e)
            continue

