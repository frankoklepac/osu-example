import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score 
from tensorflow import keras
from keras import layers


data = pd.read_csv('EXAM\data\diabetes.csv')

X = data[['Pregnancies', 'Glucose', 'BloodPressure' , 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# a)
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

# b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# c)
model.fit(X_train, y_train, epochs=150, batch_size=10)

# d)
model.save('model.h5')
