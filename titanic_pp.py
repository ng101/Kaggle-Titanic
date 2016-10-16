import pandas as pd
import os
import pickle
from sklearn import preprocessing

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
print('No of training rows:', train.shape[0])
print('No of test rows:', test.shape[0])

test_ids = test['PassengerId'] # saving for later


# Extract and append titles
# Credits for extracting Title:
# https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic/notebook
titles = train.Name.str.split(',').str[1].str.split().str[0]
train = pd.concat([train, titles.rename('Title')], axis = 1)
value_counts = titles.value_counts()
to_replace = value_counts[value_counts < 10].index
train['Title'].replace(to_replace, 'LC_TITLE', inplace=True)
titles = test.Name.str.split(',').str[1].str.split().str[0]
test = pd.concat([test, titles.rename('Title')], axis = 1)
test['Title'].replace(to_replace, 'LC_TITLE', inplace=True)

# dropping name, ticket, PassengerId as they don't seem relevant (guess)
# dropping Cabin: 687 missing values out of ~900
train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis = 1, inplace=True)

# drop couple of rows with missing Embarked values
train.dropna()

# Age has 177 missing values, filling them up with median according to titles
temp_tr = train[train['Age'].notnull()]
mean_age_mr = temp_tr.ix[temp_tr.Title == 'Mr.', 'Age'].mean()
mean_age_master = temp_tr.ix[temp_tr.Title == 'Master.', 'Age'].mean()
mean_age_mrs = temp_tr.ix[temp_tr.Title == 'Mrs.', 'Age'].mean()
mean_age_miss = temp_tr.ix[temp_tr.Title == 'Miss.', 'Age'].mean()
mean_age = temp_tr.Age.mean()

train.ix[(train.Title == 'Mr.') & (train.Age.isnull()), 'Age'] = mean_age_mr
train.ix[(train.Title == 'Master.') & (train.Age.isnull()), 'Age'] = mean_age_master
train.ix[(train.Title == 'Mrs.') & (train.Age.isnull()), 'Age'] = mean_age_mrs
train.ix[(train.Title == 'Miss.') & (train.Age.isnull()), 'Age'] = mean_age_miss
train.Age.fillna(mean_age, inplace = True)

test.ix[(test.Title == 'Mr.') & (test.Age.isnull()), 'Age'] = mean_age_mr
test.ix[(test.Title == 'Master.') & (test.Age.isnull()), 'Age'] = mean_age_master
test.ix[(test.Title == 'Mrs.') & (test.Age.isnull()), 'Age'] = mean_age_mrs
test.ix[(test.Title == 'Miss.') & (test.Age.isnull()), 'Age'] = mean_age_miss
test.Age.fillna(mean_age, inplace = True)

#test.ix[(test.Sex == 'male') & (test.Age.isnull()), 'Age'] = age_mean_male
#test.ix[(test.Sex == 'female') & (test.Age.isnull()), 'Age'] = age_mean_female

#train['Age'].fillna(0, inplace=True)
#test['Age'].fillna(0, inplace=True)

# Print number of rows again, just to verify we haven't dropped too many rows

print('No of training rows:', train.shape[0])
print('No of test rows:', test.shape[0])

print('Missing values in test data: ', sum(sum(test.isnull().values)))

# Test data has 1 missing value in Fare, fill that up with median of train data
test['Fare'].fillna(train['Fare'].mean(), inplace=True)

print('Missing values in test data: ', sum(sum(test.isnull().values)))

# We have non-null data now
# Take care of categorical data now

# change Embarked variable to one hot encoding
def dummify(tr_df, te_df, column):
    prefx = 'C_' + column + '_'
    # later we will drop column and one of new columns
    to_drop = [column]
    tr_dum = pd.get_dummies(tr_df[column]).rename(
            columns=lambda x:prefx + str(x))
    tr_df = pd.concat([tr_df, tr_dum], axis = 1)
    tr_df.drop(to_drop, inplace=True, axis=1)
    te_dum = pd.get_dummies(te_df[column]).rename(
            columns=lambda x:prefx + str(x))
    te_dum = te_dum.reindex(columns = tr_dum.columns, fill_value=0) # handle unseen data
    te_df = pd.concat([te_df, te_dum], axis = 1)
    te_df.drop(to_drop, inplace=True, axis=1)
    return tr_df, te_df

#train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
#test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
#train.drop(['SibSp'], inplace=True, axis = 1)
#test.drop(['SibSp'], inplace=True, axis = 1)

## convert SibSp, Parch to 0/1 variables
#train.ix[train.SibSp > 1, 'SibSp'] = 2
#test.ix[test.SibSp > 1, 'SibSp'] = 2
#
#train.ix[train.Parch > 0, 'Parch'] = 2
#test.ix[test.Parch > 0, 'Parch'] = 2
#
# convert sex to child, male, female
#child_age_limit = 10
#train.ix[train.Age <=child_age_limit, 'Sex'] = 'child'
#test.ix[test.Age <=child_age_limit, 'Sex'] = 'child'

elderly_age_threshold = 75
train.ix[train.Age >=elderly_age_threshold, 'Sex'] = 'elder'
test.ix[test.Age >= elderly_age_threshold, 'Sex'] = 'elder'

train, test = dummify(train, test, 'Embarked')
train, test = dummify(train, test, 'Pclass')
train, test = dummify(train, test, 'Sex')
train, test = dummify(train, test, 'Title')

Y = train['Survived']
train.drop(['Survived'], inplace=True, axis=1)


#to_keep = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'C_Embarked_C', 
#        'C_Embarked_Q', 'C_Pclass_1', 'C_Pclass_2']

#to_keep = ['Sex', 'Age', 'C_Pclass_1', 'C_Pclass_2', 'Fare']
#to_keep = ['Sex']
#train = train[to_keep]
#test = test[to_keep]

#train.drop(['Age'], axis = 1, inplace=True)
#test.drop(['Age'], axis = 1, inplace=True)

test = test[train.columns] # reindex in case some variables are not in order

print('Train columns')
print(train.head(2))
print('Test columns')
print(test.head(2))

def try_pickle(obj, filename, force = False):
    if os.path.exists(filename) and not force:
        print('%s exists, skipping..' % filename)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

TR_X = train.as_matrix()
TE_X = test.as_matrix()

min_max_scaler = preprocessing.MinMaxScaler()
TR_X = min_max_scaler.fit_transform(TR_X)
TE_X = min_max_scaler.transform(TE_X)

print(TR_X[0:2, :])
print(TE_X[0:2, :])

to_pickle = {'TR_X': TR_X, 'TR_Y': Y.as_matrix(),
        'TE_X': TE_X, 'test_ids': test_ids.as_matrix()}
try_pickle(to_pickle, 'titanic.pickle')
