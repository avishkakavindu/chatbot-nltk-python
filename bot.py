import json
import numpy as np
import nltk
import pickle
import random
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model


class ChatBot:
    IGNORED_LETTERS = ',.?!/-_'

    def __init__(self, json_file='intents.json'):
        self.intents = json.loads(open(json_file).read())
        self.lemmatizer = WordNetLemmatizer()
        self.all_words = []
        self.classes = []
        self.dataset = []

    def preprocess(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                tokenized_words = nltk.word_tokenize(pattern)
                self.all_words.extend(tokenized_words)
                # list consists of tuples that has tokenized sentences and there tag
                self.dataset.append((tokenized_words, intent['tag']))
                # add tag to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.all_words = [self.lemmatizer.lemmatize(word) for word in self.all_words if word not in self.IGNORED_LETTERS]
        self.all_words = sorted(set(self.all_words))
        self.classes = sorted(set(self.classes))

    def create_pkl_files(self):
        pickle.dump(self.all_words, open('all_words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

    def get_train_set(self):
        train_set = []
        enc_arr = [0] * len(self.classes)

        for data in self.dataset:
            bag = []
            patterns = data[0]
            patterns = [self.lemmatizer.lemmatize(word.lower()) for word in patterns]

            for word in self.all_words:
                if word in patterns:
                    bag.append(1)
                else:
                    bag.append(0)

            tmp_enc_arr = list(enc_arr)
            tmp_enc_arr[self.classes.index(data[1])] = 1
            train_set.append([bag, tmp_enc_arr])
            random.shuffle(train_set)
        return np.array(train_set)

    def xy_split(self, train_set):
        x_train = list(train_set[:, 0])
        y_train = list(train_set[:, 1])
        return x_train, y_train

    def create_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation="softmax"))

        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["accuracy"])
        print(model.summary())
        return model

    def train(self, epochs=200):
        print('here')
        self.preprocess()
        self.create_pkl_files()
        train_set = self.get_train_set()

        x_train, y_train = self.xy_split(train_set)
        input_shape = (len(x_train[0]),)
        output_shape = len(y_train[0])

        model = self.create_model(input_shape, output_shape)
        hist = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1)
        model.save('model.model')

    def clean_up(self, sentence):
        tokenized = nltk.word_tokenize(sentence)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokenized]
        return lemmatized

    def bag_of_words(self, sentence):
        all_words = pickle.load(open('all_words.pkl', 'rb'))

        cleaned_words = self.clean_up(sentence)
        bag = [0] * len(all_words)
        for word in cleaned_words:
            for idx, w in enumerate(all_words):
                if word == w:
                    bag[idx] = 1
        return np.array(bag)

    def get_predictions(self, sentence):
        bag = self.bag_of_words(sentence)
        loaded_model = load_model('model.model')

        predictions = loaded_model.predict(np.array([bag]))[0]
        ERROR_THRESHOLD = 0.5
        results = [[i, r] for i, r in enumerate(predictions) if r > ERROR_THRESHOLD]

        classes = pickle.load(open('classes.pkl', 'rb'))
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for result in results:
            return_list.append(
                {
                    'intent': classes[result[0]],
                    'probability': result[1]
                }
            )
        return return_list


if __name__ == '__main__':
    model = ChatBot()
    # model.train()

    print(model.get_predictions('hi, goodbye'))



