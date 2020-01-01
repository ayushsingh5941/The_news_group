'''Importing datasets of 20 news group'''
from sklearn.datasets import fetch_20newsgroups
'''Other imports'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''To vectorize data for unigram counts import count vectorizer'''
from sklearn.feature_extraction.text import CountVectorizer

# downloading and caching datasets
groups = fetch_20newsgroups()
print(groups.keys())

# target_names key gives the newsgroup names (Category), corrosponds to a newsgroup but encoded as an integer
print(groups['target_names'])
print(groups.target)

# using numpy Unique to get distinct values of integers

print(np.unique(groups.target)) # 0-19 representing 20 categories

#displaying data at 1st index
print(groups.data[0])
# checking category by target integer
print('Category integer', groups.target[0])
# getting name of category
print('Category name :',groups.target_names[groups.target[0]])

# counting number of words in data 1 and data 2
print('Words in data 0',len(groups.data[0]))
print('Words in data 1',len(groups.data[1]))

# Using BAG OF WORDS MODEL to find cerian words and its occurence to find potential labels

# it's good to visualize data and get fammiliar with structure, to know wha possible issues may arise
plt.figure()
sns.distplot(groups.target)

#For unigrams
cv = CountVectorizer(stop_words='english', max_features=500) # initializing countvectorizer
transformed = cv.fit_transform(groups.data) # transforming data t fit in count vectorizer, learn(s) a vocabulary dictionary of all tokens in the raw documents
#print(cv.get_feature_names())
tr = transformed.toarray().sum(axis=0)
sns.distplot(np.log(tr))
plt.xlabel('Log Count')
plt.ylabel('Frequency')
plt.title('Distribution plot of 500  word counts')
plt.show()

'''**********************************************************************************************'''

'''DATA PROCESSING'''

