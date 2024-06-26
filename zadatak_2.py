import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score 

data = pd.read_csv('EXAM\data\diabetes.csv')

X = data[['Pregnancies', 'Glucose', 'BloodPressure' , 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# a) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka
# za ucenje.
model = LogisticRegression()
model.fit(X_train, y_train)

# b) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke
# regresije.
y_pred = model.predict(X_test)

# c) Izracunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezultate.
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

# d) Izracunajte tocnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte dobivene rezultate.
print("Tocnost modela:", accuracy_score(y_test, y_pred))
print("Preciznost modela:", precision_score(y_test, y_pred))
print("Odziv modela:", recall_score(y_test, y_pred))

