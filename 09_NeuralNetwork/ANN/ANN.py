import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography = pd.get_dummies(X['Geography'], drop_first = True)
gender = pd.get_dummies(X['Gender'], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)

X.drop(['Geography', 'Gender'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 20, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 15, kernel_initializer = 'he_normal', activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 10, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.summary()

with open('model_image.jpg', 'wb') as img_file:
    img_file.write(keras.utils.model_to_dot(classifier).create_jpg())

from ann_visualizer.visualize import ann_viz
ann_viz(classifier, view = True, filename = 'model_vis.gv', title = 'Neural network')

keras.utils.plot_model(classifier, show_shapes = True, show_layer_names = True)

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_history = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100)

classifier.evaluate(X_test, y_test)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score, recall_score, precision_score
model_accuracy = accuracy_score(y_test, y_pred)
model_recall = recall_score(y_test, y_pred)
model_precision = precision_score(y_test, y_pred)

pd.DataFrame(model_history.history).plot(figsize = (5, 3))
