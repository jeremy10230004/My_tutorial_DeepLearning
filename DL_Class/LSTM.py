import Tool_DL
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, GRU, Dropout
from tensorflow.keras.models import Model


class Twitter_sentiment:
    def __init__(self):
        self.DataPath = "DataSet/twitter/training.1600000.processed.noemoticon.csv"
        self.tokenizer = None
        self.embedding_weight = None
        self.train_x, self.test_x, self.train_y, self.test_y = [], [], [], []
        self.model = None
        self.Bi = True
        self.save_name = ''

        self.padding_size = 50
        self.DropoutRate = 0.2

    @staticmethod
    def text_clear(t, stopwords=None, stemmer=None):
        temp = re.sub("(@|http:|https:|www)\S+|[^A-Za-z0-9]+", " ", t.lower()).strip()
        temp = temp.replace('#', "").replace('-', " ")
        ans = []
        if stemmer is not None:
            for i in temp.split():
                if i not in stopwords:
                    ans.append(stemmer.stem(i))
        else:
            for i in temp.split():
                if i not in stopwords:
                    ans.append(i)
        return ans

    def preprocess(self):
        df = pd.read_csv(self.DataPath, encoding='ISO-8859-1',
                         names=['target', 'ids', 'date', 'flag', 'user', 'text'])

        stop_w = Tool_DL.get_stopwords()
        df['text'] = df['text'].apply(self.text_clear, stopwords=stop_w,
                                      stemmer=SnowballStemmer('english'))  # clean data
        df['target'] = df['target'] / 4  # positive = 4 --> 1

        # word --> token
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(df['text'])
        all_x = tf.keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences(df['text']),
                                                              maxlen=self.padding_size)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(all_x, df['target'], test_size=0.2,
                                                                                random_state=77)
        print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)
        self.embedding_weight = self.word2vec()

    def word2vec(self):
        w2v_dimension = 100
        w2v_dir = Tool_DL.get_glove6B(w2v_dimension)

        embedding_weight = np.zeros((len(self.tokenizer.word_index) + 1, w2v_dimension))  # +1 is important
        for idx, word in enumerate(self.tokenizer.word_index):
            w_vector = w2v_dir.get(word)
            if w_vector is not None:
                embedding_weight[idx] = w_vector
        print(embedding_weight.shape)
        return embedding_weight

    def build_model(self, Bi=False, mode='LSTM'):
        input_layer = Input((self.padding_size,))
        emb = Embedding(self.embedding_weight.shape[0], self.embedding_weight.shape[1], weights=[self.embedding_weight],
                        input_length=self.padding_size, trainable=False)(input_layer)
        d = SpatialDropout1D(self.DropoutRate)(emb)
        if mode.upper() == 'LSTM':
            t = LSTM(32, dropout=self.DropoutRate, recurrent_dropout=self.DropoutRate)
        elif mode.upper() == 'GRU':
            t = GRU(32, dropout=self.DropoutRate, recurrent_dropout=self.DropoutRate)

        if Bi:
            l = Bidirectional(t)(d)
        else:
            l = t(d)

        d = Dense(256, activation='relu')(l)
        d = Dropout(self.DropoutRate)(d)
        d = Dense(128, activation='relu')(d)
        output_layer = Dense(1, activation='sigmoid')(d)

        self.model = Model(input_layer, output_layer)

    def training(self, optimizer="adam", loss="mse", epochs=3, batch_size=32, **kwargs):
        if kwargs == {}:
            self.model.compile(optimizer=optimizer, loss=loss)
        else:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])

        self.model.summary()
        history = self.model.fit(self.train_x, self.train_y, epochs=epochs, batch_size=batch_size, verbose=1)
        Tool_DL.show_training(history, ['loss', 'accuracy'], model_name=__name__)

    def run(self):
        print(tf.__version__)
        print(np.__version__)
        self.preprocess()
        self.build_model(Bi=self.Bi, mode='GRU')
        self.training(loss='binary_crossentropy', metrics=['accuracy'], batch_size=64)
        Tool_DL.testing(self.model, self.test_x, self.test_y)
        self.model.save(f'Result/model_{__name__}.h5')


if __name__ == '__main__':
    twitter_work = Twitter_sentiment()
    twitter_work.run()
