import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from keras.api import layers
from keras.api.models import load_model
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm

data = pd.read_csv('EXAM\data\diabetes.csv')

X = data[['Pregnancies', 'Glucose', 'BloodPressure' , 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model('model.h5')
model.summary()

# e)
score = model.evaluate(X_test, y_test)

# f)
predictions = model.predict(X_test)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

conf_matrix = confusion_matrix(y_test, predictions)
print("Matrica zabune:")
print(conf_matrix)
