#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_excel('DATASET.xlsx',engine='openpyxl')


# In[4]:


df.drop_duplicates(inplace = True)


# In[5]:


df['Disease'].value_counts()


# In[6]:


df.isna().sum()


# In[20]:


first = sns.catplot(x="Symptom_1",data=df, height=5, aspect=1.6, kind='count', order=df['Symptom_1'].value_counts().index)
plt.xticks(rotation=90)


# In[22]:


first = sns.catplot(x="Symptom_2",data=df, height=5, aspect=1.6, kind='count', order=df['Symptom_2'].value_counts().index)
plt.xticks(rotation=90)


# In[27]:


columns = df.columns
columns = columns[1:]


# In[29]:


for i in columns:
    df[i] = df[i].astype(str)


# In[30]:


df['Combined'] = df[columns].apply(lambda x: ' '.join(x), axis=1)


# In[31]:


df['Combined'][0]


# In[32]:


df = df.reset_index()


# In[33]:


del df['index']


# In[67]:


corpus = []
for i in range(0,len(df)):
    words = df['Combined'][i]
    words = words.replace('nan','')
    words = words.rstrip()
    corpus.append(words)    


# In[68]:


df['Cleared Combined'] = corpus


# In[69]:


df.shape


# In[70]:


df.head()


# In[71]:


x = df['Cleared Combined']


# In[72]:


y = df['Disease']


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state = 7,stratify = y)


# In[75]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[76]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[77]:


y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# In[78]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfvec= TfidfVectorizer()


# In[79]:


x_train = tfvec.fit_transform(x_train).toarray()
x_test = tfvec.transform(x_test).toarray()


# In[80]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

"""model_cv_rdf = GridSearchCV(estimator=RandomForestClassifier(),param_grid={"min_samples_split":[2,5,10], 'max_depth':[10,50,100],'min_samples_leaf':[1,2,4],'n_estimators':[200,1000,200]}, return_train_score = True)

model_cv_rdf.fit(x_train,y_train)

grid_rdf = pd.DataFrame(model_cv_rdf.cv_results_)

model_cv_nb = GridSearchCV(estimator=MultinomialNB(),param_grid={'alpha':[1.0,1.5,2], 'fit_prior':[True,False], 'class_prior':[0.5,0.6,0.7,None]}, return_train_score = True)

model_cv_nb.fit(x_train,y_train)

grid_nb = pd.DataFrame(model_cv_nb.cv_results_)

model_cv_knn = GridSearchCV(estimator=KNeighborsClassifier(),param_grid={'n_neighbors':[1,5,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'leaf_size':[10,20,30],'p':[1,2]}, return_train_score = True)

model_cv_knn.fit(x_train,y_train)

grid_knn = pd.DataFrame(model_cv_knn.cv_results_)"""


# In[26]:

disease_detect_rfc = RandomForestClassifier(max_depth=10,min_samples_leaf=1,min_samples_split=2,n_estimators=100).fit(x_train,y_train)


# In[27]:


y_pred_rfc = disease_detect_rfc.predict(x_test)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


print('The accuracy of Random Forest Classifier on testing set is',accuracy_score(y_test,y_pred_rfc))
print(confusion_matrix(y_test,y_pred_rfc))

# In[30]:





# In[31]:


disease_detect_nb = MultinomialNB(alpha=1.0,class_prior=None,fit_prior=False).fit(x_train,y_train)


# In[32]:


y_pred_nb = disease_detect_nb.predict(x_test)


# In[33]:


print('The accuracy of Naive Bayes on testing set is',accuracy_score(y_test,y_pred_nb))
print(confusion_matrix(y_test,y_pred_nb))

# In[34]:



# In[35]:


disease_detect_knn = KNeighborsClassifier(algorithm = 'brute', leaf_size = 30, n_neighbors = 10, p = 1, weights = 'distance').fit(x_train,y_train)


# In[36]:


y_pred_knn = disease_detect_knn.predict(x_test)


# In[37]:


print('The acurracy of k-nearest neighbors on testing set is',accuracy_score(y_test,y_pred_knn))
print(confusion_matrix(y_test,y_pred_knn))

# In[38]:


# a = tfvec.transform([str(input("Enter the symptoms you're feeling: "))]).toarray()


# In[39]:


# le.inverse_transform([disease_detect_knn.predict(a),disease_detect_nb.predict(a),disease_detect_rfc.predict(a)])


# In[40]:


import pickle


# In[41]:


# pickling the machine learning models
pickle.dump(disease_detect_rfc, open('rdf_model.pkl','wb'))
pickle.dump(disease_detect_nb, open('nb_model.pkl','wb'))
pickle.dump(disease_detect_knn, open('knn_model.pkl','wb'))


# In[46]:


# model = pickle.load(open('rdf_model.pkl','rb'))
# print(le.inverse_transform(model.predict(tfvec.transform([str("joint_pain vomiting fatigue")]).toarray())))


# In[47]:


# pickling the encoders
with open('le.pkl', 'wb') as out_tfidf:
    pickle.dump(le, out_tfidf)
with open('tfvec.pkl', 'wb') as out_le:
    pickle.dump(tfvec, out_le)

