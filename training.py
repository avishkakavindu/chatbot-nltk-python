import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


intents = json.loads(open('intents.json').read())

words = []
labels = []
docs = []
ignored_letters = [',', '.', '?', '!', '/', '-', '_']

# access intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenization
        # Ex: 'Natural Language Processing' to
        # ['Natural', 'Language', 'Processing']

        word_list = nltk.word_tokenize(pattern)
        # create a word list of tokenized pattern words
        words.extend(word_list)
        docs.append((word_list, intent['tag']))

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Lemmatizer will group together different inflected forms of a word, called lemma
# Ex: Lemmetizer will map gone, going and went to go
lemmatizer = WordNetLemmatizer()

# eliminate ignored_letters, duplicates and  sorted convert set into sorted list
words = sorted(set([lemmatizer.lemmatize(word) for word in words if word not in ignored_letters]))

labels = sorted(set(labels))

# store object data to pkl file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('labels.pkl', 'wb'))

training = []
output_empty = [0] * len(labels)
print(words)
for doc in docs:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

x_train = list(training[:, 0])
y_train = list(training[:, -1])

# model

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=4)
model.save('chatbot_model.model')



