#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:54:11 2019

@author: iurii
"""

from nltk.metrics import edit_distance
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')


df_train = pd.read_csv("input/train.csv.zip", encoding="ISO-8859-1")
df_test = pd.read_csv("input/test.csv.zip", encoding="ISO-8859-1")
attribute_data = pd.read_csv('input/attributes.csv.zip')
df_pro_desc = pd.read_csv('input/product_descriptions.csv.zip')



df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True, sort=False)

num_train = df_train.shape[0]

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

def str_stemmer_title(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')


############## apply stemming #####################
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer_title(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
############## end stemming #####################

############## building custome feature, let's build a few of them before compare which one is the best ###########
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['shared_words'] = df_all[['search_term','product_description', 'product_title']].apply(lambda row:sum([str_common_word(*row[:-1]), str_common_word(*row[1:])]), axis=1).astype(np.int64)

# training_data['frequency_digits_in_sq']=training_data.product_description.str.count("\\d+")
df_all['frequency_words_in_sq'] = df_all.product_description.str.count("\\w+").astype(np.int64)
df_all["distance_levistein"] = df_all.loc[:, ["search_term","product_title"]].apply(lambda x: edit_distance(*x), axis=1).astype(np.int64)

df_all['length in product info'] = df_all[['product_title','product_description']].apply(lambda row:sum([len(*row[:-1]), len(*row[1:])]), axis=1).astype(np.int64)

df_all = df_all.drop(['search_term','product_title','product_description'],axis=1)


df_train = df_all.iloc[:num_train]
print('df_train',df_train)
df_test = df_all.iloc[num_train:]
print('df_test',df_test)
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

#### Feature to the same scale
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.fit_transform(X_test)

rf = RandomForestRegressor(n_estimators=4, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=4, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
print("done, now you can submit the submission.csv")