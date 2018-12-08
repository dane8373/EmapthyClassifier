import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load


print(tf.VERSION)
print(tf.keras.__version__)

#load the raw data
filenames = []
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/responses.csv"

npFeatures=pd.read_csv(filename, sep=',',header=0)
#extract the labels
npLabels2=[]
for z in npFeatures.values:
    npLabels2.append(z[94])

#remove all examples that don't have our target labelled
removeInd = []
for z in range(len(npLabels2)):
    if np.isnan(npLabels2[z]):
        removeInd.append(z)
    else:
        if npLabels2[z] > 3:
            npLabels2[z] = 1
        else:
            npLabels2[z] = 0

npFeatures2 = []

for z in range(len(npFeatures)):
    if z not in removeInd:
        npFeatures2.append(npFeatures.values[z])

npLabels2 = np.delete(npLabels2,removeInd)

#Preprocessing the categorical features
for z in range(len(npFeatures2)):
    if (npFeatures2[z][73] == "never smoked"):
        npFeatures2[z][73] = 1
    elif (npFeatures2[z][73] == "tried smoking"):
        npFeatures2[z][73] = 2
    elif (npFeatures2[z][73] == "former smoker"):
        npFeatures2[z][73] = 3
    elif (npFeatures2[z][73] == "current smoker"):
        npFeatures2[z][73] = 4
    if (npFeatures2[z][74] == "never"):
        npFeatures2[z][74] = 1
    elif (npFeatures2[z][74] =="social drinker"):
        npFeatures2[z][74] = 2
    elif (npFeatures2[z][74] =="drink a lot"):
        npFeatures2[z][74] = 3
    if (npFeatures2[z][107] == "i am often early"):
        npFeatures2[z][107] = 1
    elif (npFeatures2[z][107] == "i am always on time"):
        npFeatures2[z][107] = 2
    elif (npFeatures2[z][107] == "i am often running late"):
        npFeatures2[z][107] = 3
    if (npFeatures2[z][108] == "never"):
        npFeatures2[z][108] = 1
        npFeatures2[z][94] = 1 #reusing the space that used to store the label
        #use it to store whether or not they lie pathologically
    elif (npFeatures2[z][108] == "only to avoid hurting someone"):
        npFeatures2[z][108] = 1
        npFeatures2[z][94] = 1
    elif (npFeatures2[z][108] == "sometimes"):
        npFeatures2[z][108] = 2
        npFeatures2[z][94] = 1
    elif (npFeatures2[z][108] == "everytime it suits me"):
        npFeatures2[z][108] = 3
        npFeatures2[z][94] = 2
    if (npFeatures2[z][132] == "no time at all"):
        npFeatures2[z][132] = 1
    elif (npFeatures2[z][132] == "less than an hour a day"):
        npFeatures2[z][132] = 2
    elif (npFeatures2[z][132] == "few hours a day"):
        npFeatures2[z][132] = 3
    elif (npFeatures2[z][132] == "most of the day"):
        npFeatures2[z][132] = 4
    if (npFeatures2[z][144] == "male"):
        npFeatures2[z][144] = 1
    elif (npFeatures2[z][144] == "female"):
        npFeatures2[z][144] = 2
    if (npFeatures2[z][145] == "right handed"):
        npFeatures2[z][145] = 1
    elif (npFeatures2[z][145] == "left handed"):
        npFeatures2[z][145] = 2
    if (npFeatures2[z][146] == "currently a primary school pupil"):
        npFeatures2[z][146] = 1
    elif (npFeatures2[z][146] == "primary school"):
        npFeatures2[z][146] = 2
    elif (npFeatures2[z][146] == "secondary school"):
        npFeatures2[z][146] = 3
    elif (npFeatures2[z][146] == "college/bachelor degree"):
        npFeatures2[z][146] = 4
    elif (npFeatures2[z][146] == "masters degree"):
        npFeatures2[z][146] = 5
    elif (npFeatures2[z][146] == "doctorate degree"):
        npFeatures2[z][146] = 6
    if (npFeatures2[z][147] == "no"):
        npFeatures2[z][147] = 1
    elif (npFeatures2[z][147] == "yes"):
        npFeatures2[z][147] = 2
    if (npFeatures2[z][148] == "village"):
        npFeatures2[z][148] = 1
    elif (npFeatures2[z][148] == "city"):
        npFeatures2[z][148] = 2
    if (npFeatures2[z][149] == "block of flats"):
        npFeatures2[z][149] = 1
    elif (npFeatures2[z][149] == "house/bungalow"):
        npFeatures2[z][149] = 2

#replace all nans with an average value
for z in range(len(npFeatures2[1])):
    avg = 0
    for i in range(len(npFeatures2)):
        if (np.isnan(npFeatures2[i][z])):
            continue
        else:
            avg += npFeatures2[i][z]
    avg = math.floor(avg / len(npFeatures2))
    for i in range(len(npFeatures2)):
        if (np.isnan(npFeatures2[i][z])):
            npFeatures2[i][z] = avg

npFeatures2 = np.array(npFeatures2)
npLabels2 = np.array(npLabels2)

#Do most frequent classifier
count = [0, 0]
for z in npLabels2:
    t = int(z)
    count[t-1] += 1

label = 1
if (count[0] < count[1]):
    label = 2

misses = 0
for z in npLabels2:
    t = int(z)
    if t != label:
        misses += 1

numLabels = npLabels2.size
print("\nMajority Classifier Accuracy: " + str((numLabels-misses)/numLabels) )

class_weight = {0: 1.,
                1: 2.}

#make a test train split (roughly 75-15-10 train-valid-test random split)
randomindices = np.arange(1005)
np.random.shuffle(randomindices)
test_data = npFeatures2[randomindices[905:]]
test_labels = npLabels2[randomindices[905:]]
training_set = npFeatures2[randomindices[:905]]
label_set = npLabels2[randomindices[:905]]
randomindices = np.arange(905)
training_data = training_set[randomindices[:755]]
training_labels = label_set[randomindices[:755]]
validation_data = training_set[randomindices[755:905]]
validation_labels = label_set[randomindices[755:905]] 

#Save the test data for later
np.save("Processed_Test_Data", test_data)
np.save("Processed_Test_Labels", test_labels)

#Build the tensor flow model
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
model.fit(training_data, training_labels, epochs=50, batch_size=32,
          validation_data=(validation_data, validation_labels), class_weight=class_weight)

result = model.evaluate(validation_data, validation_labels, batch_size=32, verbose=0)
theModel = model

#compare to most frequent classifier
#Do most frequent classifier
count = [0, 0]
for z in validation_labels:
    t = int(z)
    count[t-1] += 1

label = 1
if (count[0] < count[1]):
    label = 2

misses = 0
for z in validation_labels:
    t = int(z)
    if t != label:
        misses += 1
MFCacc = (len(validation_labels)-misses)/len(validation_labels)
acc = result[1]
maxDiff = acc - MFCacc
used_validation = validation_data
used_labels = validation_labels
print("\nModel accuracy - Majority Classifier accuracy:")
print(maxDiff)

#Cross validation
acc = 0
for z in range(10): 
    print("\nCross Training run #" + str(z+1) + "/10")
    randomindices = np.arange(905)
    np.random.shuffle(randomindices)
    training_data = training_set[randomindices[:755]]
    training_labels = label_set[randomindices[:755]] 
    validation_data = training_set[randomindices[755:905]]
    validation_labels = label_set[randomindices[755:905]] 
    
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
    model.fit(training_data, training_labels, epochs=50, batch_size=32,
            validation_data=(validation_data, validation_labels), class_weight=class_weight, verbose = 0)

    #inspect the individual results on the test data
    result = model.evaluate(validation_data, validation_labels, batch_size=32)
    loss = result[0]
    acc = result[1]
    print("Model accuracy and Loss")
    print(result)
    #compare to most frequent classifier
    #Do most frequent classifier
    count = [0, 0]
    for z in validation_labels:
        t = int(z)
        count[t-1] += 1

    label = 1
    if (count[0] < count[1]):
        label = 2

    misses = 0
    for z in validation_labels:
        t = int(z)
        if t != label:
            misses += 1
    MFCacc = (len(validation_labels)-misses)/len(validation_labels)
    diff = acc - MFCacc
    print("Model accuracy - Majority Classifier accuracy:")
    print(diff)
    if (diff > maxDiff):
        maxDiff = diff
        theModel = model
        used_validation = validation_data
        used_labels = validation_labels

#Print final model validation accuracy
result = theModel.evaluate(used_validation, used_labels, batch_size=32)
print("\nFinal Model accuracy and Loss")
print(result)

#interesting validation results
result = theModel.predict_classes(used_validation, batch_size=32)
print("\nConfusion Matrix")
confusionMat = confusion_matrix(used_labels, result)
print(confusionMat)
c = 0
print("\nInteresting validation results:")
comparisons = []
for z in range(len(result)):
    if (result[z][0] == 0 and used_labels[z] == 0):
        comparisons.append("Prediction #" + str(z) +":" + str(result[z][0]) + " Label:"+str(used_labels[z]))
    if (result[z][0] == 0 and used_labels[z] == 1):
        comparisons.append("Prediction #" + str(z) +":" + str(result[z][0]) + " Label:"+str(used_labels[z]))
for z in comparisons:
    print(z)
for z in range(len(result)):
    if (result[z] == used_labels[z]):
        c +=1

#Save the model and weights for later
model.save_weights('tensorModel_weights.h5')
with open('tensorModel_architecture.json', 'w') as f:
    f.write(model.to_json())



#build the random forest model
class_weight = {0: 3.,
                1: 1.}
theclf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state = 0)
maxScore = 0
minLoss = 10
maxDiff = -10

#Try 10 more iterations of train-validation splits
for z in range(10):
    print("\nCross Training run #" + str(z+1) + "/10")
    randomindices = np.arange(905)
    np.random.shuffle(randomindices)
    cross_training_data = training_set[randomindices[:755]]
    cross_training_labels = label_set[randomindices[:755]] 
    validation_data = training_set[randomindices[755:905]]
    validation_labels = label_set[randomindices[755:905]]  
    clf = RandomForestClassifier(n_estimators=500, max_depth=11, random_state = 0, class_weight = class_weight)
    clf.fit(cross_training_data, cross_training_labels)
    result = clf.predict(validation_data)
    print(result)
    c = 0
    for z in range(len(result)):
        if (result[z] == validation_labels[z]):
            c +=1
    score = c/len(result)
    print("Model accuracy")
    print(str(score))
    #compare to most frequent
    count = [0, 0]
    for z in validation_labels:
        t = int(z)
        count[t-1] += 1

    label = 1
    if (count[0] < count[1]):
        label = 2

    misses = 0
    for z in validation_labels:
        t = int(z)
        if t != label:
            misses += 1
    MFCacc = (len(validation_labels)-misses)/len(validation_labels)
    diff = score - MFCacc
    print("Model accuracy - Majority Classifier accuracy:")
    print(diff)
    if (diff > maxDiff):
        theclf = clf
        maxDiff=diff
        used_validation = validation_data
        used_labels = validation_labels

#Print the accuracy of the final model
result = theclf.predict(used_validation)
print("\nFinal Model accuracy")
c = 0
for z in range(len(result)):
    if (result[z] == used_labels[z]):
        c +=1
score = c/len(result)
print(str(score))
comparisons = []
print("\nConfusion Matrix")
confusionMat = confusion_matrix(used_labels, result)
print(confusionMat)

#Print useful validation results
print("\nInteresting validation results:")
for z in range(len(result)):
    if (result[z] == 0 and used_labels[z] == 0):
        comparisons.append("Prediction #" + str(z) +":" + str(result[z]) + " Label:"+str(used_labels[z]))
    if (result[z] == 0 and used_labels[z] == 1):
        comparisons.append("Prediction #" + str(z) +":" + str(result[z]) + " Label:"+str(used_labels[z]))
for z in comparisons:
    print(z)
featureWeights = list(zip(npFeatures, theclf.feature_importances_))

#Fix the naming error for the bin we reused
for z in featureWeights:
    if z[0] == 'Empathy':
        temp = ('Pathological Liar?', z[1])
        featureWeights.remove(z)
        featureWeights.append(temp)
featureWeights.sort(key=lambda x: x[1], reverse=True)
print("\nFeature Importances for Random Forest")
print(featureWeights)
#save the model for later
dump(theclf, 'randomForest.joblib') 