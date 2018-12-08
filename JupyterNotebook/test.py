import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load
from tensorflow.keras.models import model_from_json

test_data = np.load("Processed_Test_Data.npy")
test_labels = np.load("Processed_Test_Labels.npy")
model = tf.keras.Sequential()
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(128, activation='selu'))
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(128, activation='selu'))
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(64, activation='selu'))
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(64, activation='selu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='selu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.train.AdamOptimizer(0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy'])
model.fit(test_data, test_labels, verbose=False)
model.load_weights('tensorModel_weights.h5')
clf = load('randomForest.joblib') 

#Do most frequent classifier
count = [0, 0]
for z in test_labels:
    t = int(z)
    count[t-1] += 1

label = 1
if (count[0] < count[1]):
    label = 2

misses = 0
for z in test_labels:
    t = int(z)
    if t != label:
        misses += 1

numLabels = test_labels.size
print("\nMajority Classifier Error: " + str((numLabels-misses)/numLabels) )

class_weight = {0: 1.,
                1: 2.}

print("\nTesting Neural Network")
result = model.evaluate(test_data, test_labels, batch_size=32)
print(result)
result = model.predict_classes(test_data, batch_size=32)
confusionMat = confusion_matrix(test_labels, result)
print("\nConfusion Matrix")
print(confusionMat)

print("\nTesting Random Forest")
result = clf.predict(test_data)
c = 0
for z in range(len(result)):
    if (result[z] == test_labels[z]):
        c +=1
score = c/len(result)
print(str(score))

result = clf.predict(test_data)
confusionMat = confusion_matrix(test_labels, result)
print("\nConfusion Matrix")
print(confusionMat)
