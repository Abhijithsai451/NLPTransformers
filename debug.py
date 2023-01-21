import tensorflow as tf
from keras.layers import Embedding, Dense, BatchNormalization
from keras.utils import pad_sequences
from tensorflow import one_hot
import numpy as np

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
"""
def encode_transformer_data(data):
    onehot_repr = [one_hot(words, 50000) for words in data['clean_text']]
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=6)
    return embedded_docs"""

def positional_encoding(pos, d_model):
    PE = np.zeros((1,d_model))
    for i in range(d_model):
        if i%2 ==0:
            PE[:,i] = np.sin(pos/10000 **(i/d_model))
        else:
            PE[:,i] = np.cos(pos/10000 **((i-1)/d_model))
    return PE
max_length = 600
pes = []
for i in range(max_length):
    pes.append(positional_encoding(i,128))
pes = np.concatenate(pes, axis=0)

print(pes)

sequence = tf.random.uniform(shape= (64,6))
embed_out = []
for i in range(sequence.shape[0]):
    embed = Embedding(50000, 128)(sequence[i, :])
    embed_out.append(embed + pes[i, :])
embed_out = tf.concat(embed_out, axis=1)
print(embed_out.shape)
bot_sub_in = tf.reshape(embed_out,(64,6,128))
for i in range(5):
    # BOTTOM MULTIHEAD SUB LAYER
    bot_sub_out = []
    for j in range(bot_sub_in.shape[1]):
        values = bot_sub_in[:, :j]
        attention = [MultiHeadAttention(128, 5) for _ in range(5)][i](tf.expand_dims(bot_sub_in[:,j,:], axis =1), values)

        bot_sub_out.append(attention)

    bot_sub_out = tf.concat(bot_sub_out, axis=1)
    bot_sub_out = bot_sub_in + bot_sub_out

    bot_sub_out = [BatchNormalization() for _ in range(5)][i](bot_sub_out)
    print("bot_sub_in.shape end of loop is ",bot_sub_out.shape)
    