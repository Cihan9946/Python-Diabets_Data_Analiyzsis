import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import random
#Burada kütüphaneleri import ettik.


seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
#Tekrarlanabilirlik için rastgele tohum değerini ayarlayalım

data = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes.csv')
data.head()
#Veri setini yükleyelim


data.info()
#Veri bilgisini görelim.

# Çizgi Grafikleri
plt.figure(figsize=(10, 5))
sns.lineplot(x='Age', y='Glucose', data=data)
plt.show()

sns.heatmap(data.corr(), annot=True, fmt='0.2f')
#Korelasyon matrisini görselleştirelim

data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']] = data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']].replace(0, np.nan)
data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']] = data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']].fillna(data[['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']].mean())
#Eksik değerleri işleme

num = data[data["SkinThickness"] == 0]
num1 = data[data["BloodPressure"] == 0]
num2 = data[data["Glucose"] == 0]
num3 = data[data["Insulin"] == 0]
num4 = data[data["BMI"] == 0]
num.shape, num1.shape, num2.shape, num3.shape, num4.shape
#Sıfır değerleri içeren örneklerin sayılması

X = data.drop(columns=['Outcome'])
y = data['Outcome']
#Veriyi eğitim ve test setlerine ayıralım

ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X, y)
#Sınıf dengesizliğini ele almak için RandomOverSampler uygulayalım

sns.countplot(data=data, x=y_ros)
#Sınıf dağılımını görselleştirelim

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Eğitim verilerinde özellik ölçeklendirmesi yapalım

lgb_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=0)
lgb_classifier.fit(X_train, y_train)
#LightGBM sınıflandırıcısını oluşturalım ve eğitelim

sns.pairplot(data, diag_kind='auto', hue='Outcome', kind='scatter')
plt.show()

y_pred = lgb_classifier.predict(X_test)
#Test seti için hedef değerlerini tahmin edelim

data.hist(figsize = (20,20))

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)
#Doğruluk: 0.88

import plotly.graph_objects as go
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data['Glucose']
y = data['Insulin']
z = data['BMI']
bubble_sizes = data['BloodPressure']

ax.scatter(x, y, z, s=bubble_sizes, alpha=0.6)

ax.set_xlabel('Glucose')
ax.set_ylabel('Insulin')
ax.set_zlabel('BMI')

plt.title('Glucose vs. Insulin vs. BMI with BloodPressure')
plt.show()

for i in range(X.shape[1]):
    plt.figure()
    sns.distplot(X.iloc[:,i])
    plt.title(X.columns[i])
plt.show()


# ROC eğrisini çizelim
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) (Sensitivity)')
plt.title('ROC Curve ')
plt.legend()
plt.grid(True)
plt.show()

# AUC (Alan Altındaki Eğri) değeri
auc_score = roc_auc_score(y_test, y_pred)
print("AUC Değeri:", auc_score)

print("Karmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)
