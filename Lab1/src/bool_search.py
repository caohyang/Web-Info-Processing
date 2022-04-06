import os
import json
import nltk
import collections
from time import *
from nltk.stem import PorterStemmer

# 分析查询的相关函数
def arr_and(arr1, arr2): # 求列表的交
    p1 = 0
    p2 = 0
    result = []

    while p1 != len(arr1) and p2 != len(arr2):
        if arr1[p1] == arr2[p2]:
            result.append(arr1[p1])
            p1 += 1
            p2 += 1
        else:
            if arr1[p1] < arr2[p2]:
                p1 += 1
            else:
                p2 += 1    
    return result

def arr_or(arr1, arr2): # 求列表的并
    p1 = 0
    p2 = 0
    result = []

    while p1 != len(arr1) and p2 != len(arr2):
        if arr1[p1] == arr2[p2]:
            result.append(arr1[p1])
            p1 += 1
            p2 += 1
        else:
            if arr1[p1] < arr2[p2]:
                result.append(arr1[p1])
                p1 += 1
            else:
                result.append(arr2[p2])
                p2 += 1
    if p1 < len(arr1):
        result += arr1[p1:]
    if p2 < len(arr2):
        result += arr2[p2:]
    return result

def arr_not(arr):  # 求列表的补
    result = []
    for i in range(1, docid + 1):
        if i not in arr:
            result.append(i)
    return result

def parse_query(infix_tokens): #处理查询输入
    # Shunting Yard Algorithm 
    # 定义记号优先级
    precedence = {}
    precedence['NOT'] = 3
    precedence['AND'] = 2
    precedence['OR'] = 1
    precedence['('] = 0
    precedence[')'] = 0    

    # 将查询转换成记号栈
    output = []
    operator_stack = []
    for token in infix_tokens:
        if (token == '('):
            operator_stack.append(token)

        elif (token == ')'):
            operator = operator_stack.pop()
            while operator != '(':
                output.append(operator)
                operator = operator_stack.pop()

        elif (token in precedence):
            if (operator_stack):
                current_operator = operator_stack[-1]
                while (operator_stack and precedence[current_operator] > precedence[token]):
                    output.append(operator_stack.pop())
                    if (operator_stack):
                        current_operator = operator_stack[-1]
            operator_stack.append(token) 

        else:
            output.append(token.lower())

    while (operator_stack):
        output.append(operator_stack.pop())
    return output

def process_query(query): #使用倒排表完成检索
    # 查询的预处理
    query = query.replace('(', '( ')
    query = query.replace(')', ' )')
    query = query.split(' ')

    results_stack = []
    postfix_queue = collections.deque(parse_query(query)) # get query in postfix notation as a queue

    while postfix_queue:
        token = postfix_queue.popleft()
        result = [] 
        if (token != 'AND' and token != 'OR' and token != 'NOT'):
            stemmer = PorterStemmer()
            token = stemmer.stem(token) 
            if (token in PostingList):
                result = PostingList[token]
        elif (token == 'AND'):
            right_operand = results_stack.pop()
            left_operand = results_stack.pop()
            result = arr_and(left_operand, right_operand)   
        elif (token == 'OR'):
            right_operand = results_stack.pop()
            left_operand = results_stack.pop()
            result = arr_or(left_operand, right_operand)    
        elif (token == 'NOT'):
            right_operand = results_stack.pop()
            result = arr_not(right_operand) 
        results_stack.append(result)                        
    
    # 错误处理
    if len(results_stack) != 1: 
        print("ERROR: Invalid Query. Please check query syntax.") 
        return None
    return results_stack.pop()

# 建立倒排表
begin_time = time()
PostingList = {}
docid = 0
path = '../dataset/US_Financial_News_Articles/2018_0'

files1 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(1)) for file in files] 
files2 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(2)) for file in files] 
files3 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(3)) for file in files] 
files4 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(4)) for file in files] 
files5 = [os.path.join(root,file) for root, dirs, files in os.walk(path+str(5)) for file in files] 
all_files = files1 + files2

for file in all_files:
    # print(file)
    f = open(file,'r',encoding='UTF-8')
    data = json.load(f)
    text = data['text']
    for word in text:
        if word not in PostingList:
            PostingList[word] = [data['id']]
        else:
            if data['id'] not in PostingList[word]:
                PostingList[word].append(data['id'])
    f.close()
    docid += 1

# 按词项字典序排序，倒排表写入表格
SortedList = sorted(PostingList)
path = '../output/PostingList.dat'
f = open(path,'w',encoding='UTF-8')
for k in SortedList:
    f.write(k)
    f.write('\t')
    IndexList = PostingList[k]
    for i in range(len(IndexList)):
        f.write(str(IndexList[i]))
        f.write('\t')
    f.write('\n')
f.close()
end_time = time()
print('The running time of building the posting list: {:.4f} secs'.format(end_time - begin_time))

# 调用函数进行布尔检索
while True:
    query = input("Input your query: (AND, OR, NOT, '(' and ')' are allowed in the query.)\n")
    begin_time = time()
    results = process_query(query)
    end_time = time()
    if results is not None:
        print('Searching time: {:.4f} secs'.format(end_time - begin_time))
        print('Doc IDs: ', results)
    flag = input('Continue? (y/n) ')
    if flag == 'n' or flag == 'N':
        break