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

# a) Izgradite neuronsku mrežu sa sljedecim karakteristikama:
# - model ocekuje ulazne podatke s 8 varijabli
# - prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
# - drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
# - izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
# Ispisite informacije o mreži u terminal.
model = keras.Sequential()

model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

# b) Podesite proces treniranja mreže sa sljede´cim parametrima:
# - loss argument: cross entropy
# - optimizer: adam
# - metrika: accuracy.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i velicinom batch-a 10.
model.fit(X_train, y_train, epochs=150, batch_size=10)

# d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju ucitanog modela.
model.save('model.h5')
