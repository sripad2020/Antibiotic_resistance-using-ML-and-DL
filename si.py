import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from keras.models import  Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.metrics

lab=LabelEncoder()
data=pd.read_csv('metadata.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())

'''for i in data.columns.values:
    if len(data[i].value_counts()) <=10:
        indexs=data[i].value_counts().index
        value=data[i].value_counts().values
        plt.pie(value,labels=indexs,autopct='%1.1f%%')
        plt.title(f"pie chart {i}")
        plt.legend()
        plt.show()'''

for i in data.columns.values:
    data[i]=lab.fit_transform(data[i])

for i in data.columns.values:
    if data[i].skew() > 0:
        data[i]=data[i].fillna(data[i].mean())
    elif data[i].skew() < 0:
        data[i]=data[i].fillna(data[i].median())


'''for i in data.columns.values:
    sn.boxplot(data[i])
    plt.title(f'{i}')
    plt.show()'''


'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''
'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

correlation_matrix = data.corr()
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()'''



'''for i in data.columns.values:
    sn.boxplot(data[[f'{i}']])
    plt.show()'''

'''for i in data.columns.values:
    sn.violinplot(data[[f'{i}']])
    plt.show()'''

'''
sn.pairplot(data)
plt.show()

sn.pairplot(data,hue='pen_sr')
plt.show()'''

'''for i in data.columns.values:
    for j in data.columns.values:
        sn.scatterplot(data=data,x=i,y=j,hue='pen_sr')
        plt.show()'''


for i in data.columns.values:
    print(data[i].value_counts())


x=data[['Tetracycline', 'Penicillin','azm_mic','tet_sr','log2_tet_mic','log2_pen_mic',
        'tet_mic','pen_mic','Beta.lactamase','Azithromycin','cip_mic',
        'log2_azm_mic','log2_cip_mic', 'cip_sr','Ciprofloxacin']]
        
y=data[['pen_sr']]

lazy=LazyClassifier()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)
models,prediction=lazy.fit(x_train,x_test,y_train,y_test)
print(prediction)

lr = LogisticRegression(max_iter=35)
lr.fit(x_train, y_train)
print('The logistic regression: ', lr.score(x_test, y_test))

tree = DecisionTreeClassifier(criterion='gini', max_depth=1.2)
tree.fit(x_train, y_train)
print('Dtree ', tree.score(x_test,y_test))

rforest = RandomForestClassifier(criterion='gini',max_depth=1.0)
rforest.fit(x_train, y_train)
print('The random forest: ', rforest.score(x_test, y_test))

X = data[['Tetracycline', 'Penicillin', 'azm_mic', 'tet_sr', 'log2_tet_mic', 'log2_pen_mic',
          'tet_mic', 'pen_mic', 'Beta.lactamase', 'Azithromycin', 'cip_mic',
          'log2_azm_mic', 'log2_cip_mic', 'cip_sr', 'Ciprofloxacin']]

Y =pd.get_dummies(data['pen_sr'])
x_Train,x_Test,y_Train,y_Test=train_test_split(X,Y,test_size=0.5)

models=Sequential()
models.add(Dense(units=X.shape[1],input_dim=X.shape[1],activation=keras.activations.relu))
models.add(Dense(units=X.shape[1],activation=keras.activations.tanh))
models.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=X.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=X.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=Y.shape[1],activation=keras.activations.softmax))
models.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics='accuracy')
histo=models.fit(x_Train,y_Train,batch_size=25,epochs=100,validation_data=(x_Test,y_Test),validation_batch_size=35)
plt.plot(histo.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(histo.history['loss'], label='loss', marker='o', color='darkblue')
plt.plot(histo.history['val_loss'],label="VAL_Loss",marker='x',color='magenta')
plt.plot(histo.history['val_accuracy'],label="VAL_ACC",marker='x',color='yellow')
plt.title('Training Vs  Validation accuracy with adam adam')
plt.legend()
plt.show()

models1=Sequential()
models1.add(Dense(units=X.shape[1],input_dim=X.shape[1],activation=keras.activations.relu))
models1.add(Dense(units=X.shape[1],activation=keras.activations.tanh))
models1.add(Dense(units=X.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=X.shape[1],activation=keras.activations.softmax))
models1.add(Dense(units=X.shape[1],activation=keras.activations.softmax))
models1.add(Dense(units=Y.shape[1],activation=keras.activations.softmax))
models1.compile(optimizer='rmsprop',loss=keras.losses.categorical_crossentropy,metrics='accuracy')
hist=models1.fit(x_Train,y_Train,batch_size=25,epochs=75,validation_data=(x_Test,y_Test),validation_batch_size=35)
plt.plot(hist.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(hist.history['loss'], label='loss', marker='o', color='darkblue')
plt.plot(hist.history['val_loss'],label="VAL_Loss",marker='x',color='magenta')
plt.plot(hist.history['val_accuracy'],label="VAL_ACC",marker='x',color='yellow')
plt.title('Training Vs  Validation accuracy with adam rmsprop')
plt.legend()
plt.show()