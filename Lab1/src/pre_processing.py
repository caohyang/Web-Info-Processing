import os
import nltk
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = PorterStemmer()
path = '../dataset/US_Financial_News_Articles' 
all_files = [os.path.join(root,file) for root, dirs, files in os.walk(path) for file in files] 
counttext = 0

for file in all_files:
    f = open(file,'r+',encoding='UTF-8')
    data = json.load(f)
    tmptext = data['text']
    f.close()

    #分词，词根化，停用词
    tmptext = [stemmer.stem(word) for word in word_tokenize(tmptext)]
    clean_tokens= []
    sr=stopwords.words('english')
    for token in tmptext:
        if (token not in sr) and token.isalpha(): 
            clean_tokens.append(token)
    
    #创建json文件，保留docid、title和text属性并存回原地址
    counttext += 1
    jsontext = {}
    jsontext['id'] = counttext
    jsontext['title'] = data['title']
    jsontext['text'] = clean_tokens
    f = open(file,'w',encoding='UTF-8')
    f.write(json.dumps(jsontext))
    f.close()