import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from scipy import stats 

# Rastgele tohumları ayarla
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# Veriyi yükle
df = pd.read_csv('/kaggle/input/wine-quality-dataset/WineQT.csv')

# Gereksiz sütunları çıkar
df.drop('Id', axis=1, inplace=True)

# Etiketleri yeniden sınıflandır
df["quality"] = df["quality"].apply(lambda x: 1 if x > 5 else 0)
x = df.drop(['quality'], axis=1)
y = df['quality']

# Veriyi train ve test setlere ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, shuffle=False, test_size=0.3)


# Modeli oluştur
classifier = Sequential()
classifier.add(Dense(units=256, kernel_initializer='uniform', activation='relu', input_dim=len(x.columns), name="layer1"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=72, kernel_initializer='uniform', activation='relu', name="layer2"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=36, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Modeli derle
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
classifier.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1, validation_data=(x_test, y_test))

# Modeli değerlendir
train_score, train_acc = classifier.evaluate(x_train, y_train, batch_size=256)
print('Train score:', train_score)
print('Train accuracy:', train_acc)



test_score, test_acc = classifier.evaluate(x_test, y_test, batch_size=10)
print('Test score:', test_score)
print('Test accuracy:', test_acc)


#Korelasyon Isı Haritası ve Matrisi:
sns.set(font_scale=1)
fig, ax = plt.subplots(2, 2, figsize=(15, 13))
fig.subplots_adjust(hspace=.5, wspace=.3)
ax = ax.flatten()
threshold = 0.28

# Train verileri için korelasyon haritası
corrs = df.corr()
np.fill_diagonal(corrs.values, np.nan)
sns.heatmap(corrs, cmap="bwr", annot=True, fmt='.2f', linewidths=.05,
            annot_kws={"fontsize": 11}, ax=ax[0], vmin=-1, vmax=1)

# Korelasyon matrisinin düzeltilmesi ve görselleştirilmesi
for k, v in corrs.items():
    v[(v < threshold) & (v > 0)] = 0

    #Korelasyon Isı Haritası ve Matrisi:



sns.set(font_scale=1)
fig, ax = plt.subplots(2, 2, figsize=(15, 13))
fig.subplots_adjust(hspace=.5, wspace=.3)
ax = ax.flatten()
threshold = 0.28

# Train verileri için korelasyon haritası
corrs = df.corr()
np.fill_diagonal(corrs.values, np.nan)
sns.heatmap(corrs, cmap="bwr", annot=True, fmt='.2f', linewidths=.05,
            annot_kws={"fontsize": 11}, ax=ax[0], vmin=-1, vmax=1)

# Korelasyon matrisinin düzeltilmesi ve görselleştirilmesi
for k, v in corrs.items():
    v[(v < threshold) & (v > 0)] = 0
    v[(v > -threshold) & (v < 0)] = 0
corrs.replace(0, np.nan, inplace=True)
corrs = corrs.where(np.triu(np.ones(corrs.shape)).astype(bool))
sns.heatmap(corrs, cmap="bwr", annot=True, fmt='.2f', linewidths=.05,
            annot_kws={"fontsize": 11}, ax=ax[2], vmin=-1, vmax=1)

# Test verileri için korelasyon haritası
corrs = x_test.corr()
corrs['quality'] = np.nan
corrs.loc['quality'] = np.nan
np.fill_diagonal(corrs.values, np.nan)
sns.heatmap(corrs, cmap="bwr", annot=True, fmt='.2f', linewidths=.05,
            annot_kws={"fontsize": 11}, ax=ax[1], vmin=-1, vmax=1)

# Korelasyon matrisinin düzeltilmesi ve görselleştirilmesi
for k, v in corrs.items():
    v[(v < threshold) & (v > 0)] = 0
    v[(v > -threshold) & (v < 0)] = 0
corrs.replace(0, np.nan, inplace=True)
corrs = corrs.where(np.triu(np.ones(corrs.shape)).astype(bool))
sns.heatmap(corrs, cmap="bwr", annot=True, fmt='.2f', linewidths=.05,
            annot_kws={"fontsize": 11}, ax=ax[3], vmin=-1, vmax=1)

fig.suptitle('Combined train and original vs test datasets', fontsize=20, fontweight='bold', x=.45)
plt.show()
#Bu kod, veri çerçevesindeki özelliklerin korelasyonunu görselleştirir. İlk olarak eğitim verileri için, ardından test verileri için iki ayrı korelasyon haritası ve düzeltilmiş korelasyon matrisi oluşturur.


#Kutu Grafiği (Box Plot):
plt.figure(figsize=(20, 10))
sns.boxplot(x='pH', y='alcohol', data=df)
plt.xlabel('Quality')
plt.ylabel('Alcohol Content')
plt.title('Box Plot of Alcohol Content by Wine Quality')
plt.show()

#Sahalama Grafiği (Scatter Plot):
plt.figure(figsize=(10,10))
sns.scatterplot(x = df['alcohol'], y = df['volatile acidity'], hue = df['quality'])
sns.regplot(x = df['quality'], y = df['volatile acidity'])
plt.title('Quality and volatile acidity')

#Kerteriz Grafiği (Kernel Density Estimation - KDE):
fig, ax = plt.subplots(3, 4, figsize=(20, 15))
ax = ax.flatten()

total_col = x.columns
for idx, col in enumerate(total_col):
    if col != 'quality':
        sns.kdeplot(data=x, x=col, fill=True, ax=ax[idx], alpha=0.1)
        sns.kdeplot(data=x_train, x=col, fill=True, ax=ax[idx], alpha=0.1)
        sns.kdeplot(data=x_test, x=col, fill=True, ax=ax[idx], alpha=0.1)
    else:
        sns.histplot(data=df, x=col, ax=ax[idx], binwidth=0.3)
        sns.histplot(data=x_train, x=col, ax=ax[idx])

    ax[idx].set(title=col)
    ax[idx].set(xlabel=None)
    ax[idx].set(ylabel=None)

labels = ['Original', 'Train', 'Test']
fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=20, ncol=3)
plt.show()

#Üç Boyutlu Sahalama Grafiği (3D Scatter Plot):

import plotly.express as px

# Sütunları seç
x_column = "fixed acidity"
y_column = "volatile acidity"
z_column = "citric acid"

# 3D saçılma grafiğini oluştur
fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column, color="quality", opacity=0.7)

fig.update_layout(scene=dict(xaxis_title=x_column, yaxis_title=y_column, zaxis_title=z_column))
fig.show()

#Uni-Variate Analiz ve Görselleştirmeler:

col_list = df.columns

fig, ax = plt.subplots(len(col_list), 2, figsize=(12, 6 * len(col_list)))
for index, i in enumerate(col_list):
    sns.distplot(df[i], ax=ax[index, 0], color='green')
    sns.boxplot(data=df, x=i, ax=ax[index, 1], color='yellow')
    
fig.tight_layout()
plt.suptitle("Uni-Variate Analysis of Continuous Variables", fontsize=16)
plt.show()