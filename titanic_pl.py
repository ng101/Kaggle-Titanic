import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
train['Sex'] = (train['Sex'] == 'male').astype(int)
train['Embarked'] = train['Embarked'].astype('category').cat.codes
train['Cabin'] = train['Cabin'].astype('category').cat.codes
#train.plot(x='Fare', y='Survived', kind='scatter')
#train['Survived'].hist(by=train['Embarked'])
train.groupby('Survived').hist()
test = pd.read_csv('test.csv')
test['Sex'] = (test['Sex'] == 'male').astype(int)
test['Embarked'] = test['Embarked'].astype('category').cat.codes
test['Cabin'] = test['Cabin'].astype('category').cat.codes

test.hist()
plt.show()
