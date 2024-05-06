import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EXAM\data\diabetes.csv')

# a)

print("Mjerenja su izvresna na:", data.shape[0], "pacijenata")

# b)
data['Age'].isnull()
data = data.dropna()
data=data.drop_duplicates()
print("Mjerenja su izvresna na:", data.shape[0], "pacijenata")

# c)
plt.scatter(data['Age'], data['BMI'])
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age vs BMI')
plt.show()

# d)
min_bmi = data['BMI'].min()
max_bmi = data['BMI'].max()
mean_bmi = data['BMI'].mean()

print("BMI: min=", min_bmi, "max=", max_bmi, "mean=", mean_bmi)

# e)
min_bmi_diabetes = data[data['Outcome'] == 1]['BMI'].min()
max_bmi_diabetes = data[data['Outcome'] == 1]['BMI'].max()
mean_bmi_diabetes = data[data['Outcome'] == 1]['BMI'].mean()

print("BMI za pacijente sa dijabetesom: min=", min_bmi_diabetes, "max=", max_bmi_diabetes, "mean=", mean_bmi_diabetes)

min_bmi_no_diabetes = data[data['Outcome'] == 0]['BMI'].min()
max_bmi_no_diabetes = data[data['Outcome'] == 0]['BMI'].max()
mean_bmi_no_diabetes = data[data['Outcome'] == 0]['BMI'].mean()
print("BMI za pacijente bez dijabetesa: min=", min_bmi_no_diabetes, "max=", max_bmi_no_diabetes, "mean=", mean_bmi_no_diabetes)

