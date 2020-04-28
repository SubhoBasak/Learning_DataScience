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

#import keras
#
#model = keras.models.Sequential()
#model.add(keras.layers.Input(input_shape = [11]))
#model.add(keras.layers.Dense(units = 10, activation = 'relu', kernel_initializer = 'he_normal'))
#model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.Dense(untis = 7, activation = 'relu', kernel_initializer = 'he_uniform'))
#model.add(keras.layers.Dropout(0.15))
#model.add(keras.layers.Dense(units = 5, activation = 'relu', kernel_initializer = 'he_normal'))
#model.add(keras.layers.Dropout(0.1))
#model.add(keras.layers.Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
#
#model.summary()
#
#from ann_visualizer.visualize import ann_viz
#ann_viz(model, view = True, filename = 'ann_digram.gv', title = 'ann for churn modelling')
#
#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization
from keras.activations import relu, sigmoid

def create_model(layers, activation):
    model = Sequential()
    
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim = X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))

    model.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = create_model, verbose = 0)
layers = [(20,), (20, 10), (30, 20, 10)]

activations = ['sigmoid', 'relu']

param_grid = dict(layers = layers, activation = activations, batch_size = [128, 256], epochs = [30])
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3)

grid_result = grid.fit(X_train, y_train)

[grid_result.best_score_, grid_result.best_params_]