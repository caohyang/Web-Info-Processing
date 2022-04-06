from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import nltk
import json
import pandas as pd
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
path = '../dataset/US_Financial_News_Articles/2018_0'
files1 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(1)) for file in files] 
files2 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(2)) for file in files] 
all_files = files1

corpus=[]
for file in all_files:
    # print(file)
    f = open(file, 'r+', encoding='UTF-8')
    strlist = json.load(f)['text']
    str1 = ''
    for word in strlist:
        str1 += word + ' '
    corpus.append(str1)
    f.close()

tfidf_v = TfidfVectorizer(max_features=1000)
X = tfidf_v.fit_transform(corpus)
Y = csr_matrix(X.toarray())
data = pd.DataFrame(Y)
writer = pd.ExcelWriter('../output/tfidf.xlsx')
data.to_excel(writer)
writer.save()
writer.close()

query = list(input('Input the semantic query: '))
query_fit = tfidf_v.transform(query)
print(query_fit.toarray())
