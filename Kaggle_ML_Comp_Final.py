# In[1]:
##downloaded the three files
import pandas as pd
import numpy as np
tr_df = pd.read_csv('train.csv') #loading the train data


# In[2]:

new_title = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "theCountess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

def add_title(df):
    df['title'] = df['Name'].apply(lambda x: x.split(",")[1])
    df['title'] = df['title'].apply(lambda x: x.split(".")[0])
    df.title = df.title.str.replace(' ', '')
    
tr_df
add_title(tr_df)
tr_df['title'] = tr_df['title'].apply(lambda x: new_title[x])
tr_df['title'].value_counts()


# In[3]:


# group by Sex, Pclass, and Title 
grouped = tr_df.groupby(['Sex','Pclass', 'title'])  
# view the median Age by the grouped features 
grouped.Age.median()


# In[4]:


#data imputation:
tr_df['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))
tr_df['Embarked']=tr_df['Embarked'].fillna(tr_df['Embarked'].mode().iloc[0])
tr_df['Family']=tr_df['SibSp']+tr_df['Parch']


# In[5]:


#The chances of survival dropped drastically if someone traveled with more than 2 siblings or spouse.
tr_df['Is_Alone'] = tr_df['Family'] == 0


# In[6]:


tr_df
tr_df.drop(["PassengerId", "Name", 'Ticket', 'SibSp', 'Parch', 'Cabin', 'title'], axis = 1, inplace=True)


# In[7]:


tr_df


# In[8]:


#categorical transforms:
def sex_to_num(x):
    if x == "female":
        return 1
    elif x== 'male':
        return -1
    
tr_df['Sex'] = tr_df['Sex'].apply(sex_to_num)


# In[9]:


#categorical transforms:
def is_alone_to_num(x):
    if x == False:
        return -1
    elif x== True:
        return 1
    
tr_df['Is_Alone'] = tr_df['Is_Alone'].apply(is_alone_to_num)


# In[10]:


one_hot_encoded = pd.get_dummies(tr_df['Embarked'], prefix='Embarked')

tr_df = pd.concat([tr_df, one_hot_encoded], axis=1)

tr_df.drop('Embarked', axis=1, inplace=True)


# In[11]:


tr_df


# In[12]:


#before data normalization, we must check distributions for outliers (don't want to divide by max=outlier):
print(tr_df.boxplot(column = 'Age', by='Survived'))
print(tr_df.boxplot(column = 'Fare', by='Survived'))


# In[13]:


#normalize data: use 'RobustScaler' to, well, be robust against outliers!
#idea obtained from Geeks for Geeks: https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
from sklearn.preprocessing import RobustScaler
# Create a RobustScaler object
scaler = RobustScaler()
# Fit the scaler on the "Fare" column
scaler.fit(tr_df[['Fare']])
# Transform the "Fare" column using the scaler
tr_df['Fare_scaled'] = scaler.transform(tr_df[['Fare']])


# In[14]:


scaler = RobustScaler()
# Fit the scaler on the "Fare" column
scaler.fit(tr_df[['Age']])
# Transform the "Fare" column using the scaler
tr_df['Age_scaled'] = scaler.transform(tr_df[['Age']])


# In[15]:


#categorical transforms:
def Pclass_to_num(x):
    if x == 3:
        return 1
    elif x== 2:
        return 0
    elif x==1:
        return -1
tr_df['Pclass'] = tr_df['Pclass'].apply(Pclass_to_num)


# In[18]:


###our model is now categorically transformed, filled nulls, and normalized. 
##The data pre-procc is over -- onto building our model!


# In[16]:


tr_df.drop('Fare', axis=1, inplace = True)
tr_df.drop('Age', axis=1, inplace = True)


# In[ ]:





# In[17]:


tr_df


# In[18]:


#below ran!!
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD ##new SGD right here!!
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import initializers


# In[19]:


#X is training data input; Y is training data labels
X = tr_df.drop(['Survived'], axis=1).to_numpy()
Y = tr_df['Survived'].to_numpy()
X.shape


# In[66]:


from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(units = 11, input_shape=(X.shape[1],), activation ='relu', use_bias = True, bias_initializer='zeros'),  # first hidden layer
    Dropout(0.15),
    Dense(units = 9, activation = "relu", use_bias=True, bias_initializer="zeros"),                   # second hidden layer
    Dropout(0.15),
    Dense(units = 5, activation = "relu", use_bias=True, bias_initializer="zeros"),
    Dropout(0.15),
    Dense(units = 1, activation = "sigmoid")                  # output layer
])


# In[67]:


model.summary()


# In[68]:


model.compile(optimizer = SGD(learning_rate = 0.01, momentum=0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[69]:


model.fit(x=X, y=Y, validation_split=0.1, batch_size = 32, 
          epochs = 200, shuffle = True, verbose = 2)


# In[70]:


##data_changes (make this a function!)
test_df = pd.read_csv('test.csv')
#data imputation:
add_title(test_df)
test_df['title'] = test_df['title'].apply(lambda x: new_title[x])
grouped = test_df.groupby(['Sex','Pclass', 'title'])  
test_df['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))

test_df['Embarked']=test_df['Embarked'].fillna(test_df['Embarked'].mode().iloc[0])
test_df['Family']=test_df['SibSp']+test_df['Parch']

test_df['Is_Alone'] = test_df['Family'] == 0
test_df.drop(["PassengerId", "Name", 'Ticket', 'SibSp', 'Parch', 'Cabin', 'title'], axis = 1, inplace=True)

def sex_to_num(x):
    if x == "female":
        return 1
    elif x== 'male':
        return -1
    
test_df['Sex'] = test_df['Sex'].apply(sex_to_num)

#categorical transforms:
def is_alone_to_num(x):
    if x == False:
        return -1
    elif x== True:
        return 1
    
test_df['Is_Alone'] = test_df['Is_Alone'].apply(is_alone_to_num)

one_hot_encoded = pd.get_dummies(test_df['Embarked'], prefix='Embarked')

test_df = pd.concat([test_df, one_hot_encoded], axis=1)

test_df.drop('Embarked', axis=1, inplace=True)

# # Create a RobustScaler object
scaler = RobustScaler()
# # Fit the scaler on the "Fare" column
scaler.fit(test_df[['Fare']])
# # Transform the "Fare" column using the scaler
test_df['Fare_scaled'] = scaler.transform(test_df[['Fare']])

scaler = RobustScaler()
# # Fit the scaler on the "Fare" column
scaler.fit(test_df[['Age']])
# # Transform the "Fare" column using the scaler
test_df['Age_scaled'] = scaler.transform(test_df[['Age']])

def Pclass_to_num(x):
    if x == 3:
        return 1
    elif x== 2:
        return 0
    elif x==1:
        return -1
test_df['Pclass'] = test_df['Pclass'].apply(Pclass_to_num)


test_df.drop('Fare', axis=1, inplace = True)
test_df.drop('Age', axis=1, inplace = True)


# #why does fare_scaled have a missing value?

# In[71]:


test_df


# In[72]:


test_samples = test_df.to_numpy()


# In[73]:


predictions = model.predict(x=test_samples)     # Run our points through our best model


# In[74]:


print(predictions)


# In[75]:


rounded_predictions = np.rint(predictions)   # Pick the high prob in each prediction 
print(rounded_predictions)


# In[76]:


submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': rounded_predictions[:,0]
})
submission.sort_values('PassengerId', inplace=True)


# In[77]:


np.savetxt('ElyHahami_Titanic_Predictions_FOURTY-SIX.csv', rounded_predictions)


# In[ ]:


#note: copy FOURTY-FOUR was 78.9% accuracy copy!
#note: copy FOURTY-SIX was 79.4% accuracy copy

