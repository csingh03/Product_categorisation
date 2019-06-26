
# coding: utf-8

# In[12]:

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

df=pd.read_csv('Train.csv')
df1=pd.read_csv('Test.csv')

X_train_all=df['Item_Description']
y_train_all=df['Product_Category']

count_vect = CountVectorizer()
X_train_all_counts = count_vect.fit_transform(X_train_all)
tfidf_transformer_all = TfidfTransformer()
X_train_tfidf_all = tfidf_transformer_all.fit_transform(X_train_all_counts)

clf2 = LinearSVC().fit(X_train_tfidf_all, y_train_all)


# In[13]:

predicted_result=clf2.predict(count_vect.transform(df1.Item_Description))


# In[14]:

output=pd.DataFrame({'Inv_Id':df1.Inv_Id,'Product_Category':predicted_result})


# In[16]:

output.to_csv('submission.csv', index=False)


# In[ ]:



