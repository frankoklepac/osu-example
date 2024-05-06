import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EXAM\data\diabetes.csv')

# a) Na temelju veličine polja data, na koliko osoba su izvrsena mjerenja?

print("Mjerenja su izvresna na:", len(data), "pacijenata")

# b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne mase (BMI)? Obrisite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
data['BMI'].isnull()
data = data.dropna()
data=data.drop_duplicates()
print("Mjerenja su izvresna na:", len(data), "pacijenata")

# c) Prikazite odnos dobi i indeksa tjelesne mase (BMI) osobe pomocu scatter dijagrama.
# Dodajte naziv dijagrama i nazive osi s pripadajucim mjernim jedinicama. Komentirajte odnos dobi i BMI prikazan dijagramom.
plt.scatter(data['Age'], data['BMI'])
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age vs BMI')
plt.show()

# d) Izracunajte i ispisite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne
# mase (BMI) u ovom podatkovnom skupu.
min_bmi = data['BMI'].min()
max_bmi = data['BMI'].max()
mean_bmi = data['BMI'].mean()

print("BMI: min=", min_bmi, "max=", max_bmi, "mean=", mean_bmi)

# e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
# kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene
# vrijednosti.
min_bmi_diabetes = data[data['Outcome'] == 1]['BMI'].min()
max_bmi_diabetes = data[data['Outcome'] == 1]['BMI'].max()
mean_bmi_diabetes = data[data['Outcome'] == 1]['BMI'].mean()

print("BMI za pacijente sa dijabetesom: min=", min_bmi_diabetes, "max=", max_bmi_diabetes, "mean=", mean_bmi_diabetes)

min_bmi_no_diabetes = data[data['Outcome'] == 0]['BMI'].min()
max_bmi_no_diabetes = data[data['Outcome'] == 0]['BMI'].max()
mean_bmi_no_diabetes = data[data['Outcome'] == 0]['BMI'].mean()
print("BMI za pacijente bez dijabetesa: min=", min_bmi_no_diabetes, "max=", max_bmi_no_diabetes, "mean=", mean_bmi_no_diabetes)

