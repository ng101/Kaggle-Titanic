import pickle
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import optimizers
from keras.regularizers import l2, activity_l2

data = {}
filename = 'titanic.pickle'

with open(filename, 'r') as f:
    data = pickle.load(f)

TR_X = data['TR_X']
TR_Y = np_utils.to_categorical(data['TR_Y'])
TE_X = data['TE_X']

permutation = np.random.permutation(TR_X.shape[0])
TR_X = TR_X[permutation, :]
TR_Y = TR_Y[permutation, :]

print(TR_X[0:2, :])
print(TE_X[0:2, :])

test_ids = data['test_ids']

model = Sequential()
model.add(Dense(output_dim=256, activation='relu', input_dim = TR_X.shape[1],
    W_regularizer=l2(0.008)))
model.add(Dense(256, activation='relu', W_regularizer=l2(0.008)))
model.add(Dense(256, activation='relu', W_regularizer=l2(0.008)))
model.add(Dense(TR_Y.shape[1], W_regularizer=l2(0.008)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.Adam(), metrics=['accuracy'])

#model.compile(loss='mean_squared_error',
#        optimizer=optimizers.Adagrad(), metrics=['accuracy'])
# fit
model.fit(TR_X, TR_Y, nb_epoch=1000, batch_size = TR_X.shape[0],
        shuffle = True, validation_split=0.4)

# evaluate
tr_score = model.evaluate(TR_X, TR_Y, batch_size = 32)
print('Train score: ', tr_score)

# predict
pred = model.predict_classes(TE_X)

print(test_ids.shape)
print(pred.shape)

# Dump
df = pd.DataFrame({'PassengerId': test_ids, 'Survived': pred})
df.to_csv('solution.csv', index=False)
