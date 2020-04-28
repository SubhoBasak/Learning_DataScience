import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
import keras
from keras.layers import Dense, Dropout, Activation
from keras.activations import relu, sigmoid
from keras.wrappers.scikit_learn import KerasClassifier

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

geography = pd.get_dummies(X['Geography'], drop_first = True)
gender = pd.get_dummies(X['Gender'], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)

X.drop(['Gender', 'Geography'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

def create_model(layers):
    model = keras.models.Sequential()
    
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(units = nodes, input_dim = X_train.shape[1], activation = 'relu'))
#            model.add(Activation(activation))
            model.add(Dropout(rate = 0.3))
        else:
            model.add(Dense(units = nodes, activation = 'relu'))
#            model.add(Activation(activation))
            model.add(Dropout(rate = 0.3))
    
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

model = KerasClassifier(build_fn = create_model, verbose = 0)

layer_param = [(10, 5), (15, 10, 5), (20, 15, 10, 5)]
batch_size = [30]
epochs = [30]

param_grid = dict(layers = layer_param, batch_size = batch_size, epochs = epochs)

grid_model = GridSearchCV(model, param_grid = param_grid, cv = 3)
grid_model.fit(X_train, y_train)

print('best parameters : ', grid_model.best_params_)
print('best score : ', grid_model.best_score_)

model = grid_model.best_estimator_
y_pred = model.predict(X_test)

print('accuracy score : ', accuracy_score(y_test, y_pred))
print('recall score : ', recall_score(y_test, y_pred))
print('precision scsore : ', precision_score(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)

y_pred = model.predict_proba(X_test)[:, 1]

precision, recall, threshold = precision_recall_curve(y_test, y_pred)

plt.figure(figsize = (5, 3))
plt.title('precision recall curve with threshold', fontsize = 10, color = 'orange')
plt.plot(threshold, precision[:-1], label = 'precision')
plt.plot(threshold, recall[:-1], label = 'recall')
plt.xlabel('threshold', fontsize = 10, color = 'blue', rotation = 0)
plt.ylabel('score', fontsize = 10, color = 'green', rotation = 0)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize = (5, 3))
plt.title('precision recall curve', fontsize = 10, color = 'orange')
plt.plot(recall, precision)
plt.xlabel('recall', fontsize = 10, color = 'blue', rotation = 0)
plt.ylabel('precision', fontsize = 10, color = 'green', rotation = 0)
plt.grid(True)
plt.show()

fpr, tpr, threshold = roc_curve(y_test, y_pred)

plt.figure(figsize = (5, 3))
plt.title('roc curve threshold', fontsize = 10, color = 'orange')
plt.plot(threshold, fpr, label = 'false positive rate')
plt.plot(threshold, tpr, label = 'true positive rate')
plt.xlabel('threshold', fontsize = 10, color = 'blue', rotation = 0)
plt.ylabel('score', fontsize = 10, color = 'green', rotation = 0)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize = (5, 3))
plt.title('roc curve')
plt.plot(fpr, tpr)
plt.xlabel('false positive rate', fontsize = 10, color = 'blue', rotation = 0)
plt.ylabel('true positive rate', fontsize = 10, color = 'green', rotation = 0)
plt.grid(True)
plt.show()