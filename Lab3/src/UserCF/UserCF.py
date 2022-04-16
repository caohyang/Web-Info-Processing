
# https://github.com/mercurialgh/collaborative-filtering
import math
from pathlib import Path

class userCF(object):
    def __init__(self):
        self.train = dict()
        self.test = dict()
        self.usersim = dict()
        self.aver_rate = dict()
        self.div = dict()     # 相似度公式中分母的系数

    def initData(self):
        with open(Path(__file__).parent.parent.parent.joinpath('DoubanMusic.txt'),'r') as f:
            content = f.readlines()
            for line in content:
                data = line.strip().split("\t")
                data[0] = int(data[0])

                self.train[data[0]] = dict()
                self.test[data[0]] = dict()
                tmp = dict()  # item and rating
                for index in range(1, len(data)):
                    tmplist = data[index].split(',')
                    tmp[int(tmplist[0])] = int(tmplist[1])
                self.train[data[0]] = tmp  
                
                tmp = dict()
                tmplist = data[len(data)-1].split(',')
                tmp[int(tmplist[0])] = int(tmplist[1])
                self.test[data[0]] = tmp
        
        for user in self.train.keys():
            item_count = 0
            average_rate = 0
            for item, rate in self.train[user].items():
                if rate > -1:
                    item_count += 1
                    average_rate += rate
            if item_count > 0:
                self.aver_rate[user] = average_rate / item_count
            else:
                self.aver_rate[user] = 3    # a middle score 
                
        
        for user in self.train.keys():
            coeffiency = 0
            for item, rate in self.train[user].items():
                if rate > -1:
                    coeffiency += (rate - self.aver_rate[user]) ** 2
            self.div[user] = math.sqrt(coeffiency)

    def userSimilarity_3(self, train):
        '''Overwrite the function.
        item_user = dict()
        for user,items in train.items():
            for item in items.keys():
                if item not in item_user:
                    item_user[item] = set()
                item_user[item].add(user)
        
        # calculate co-rated items for users
        N = dict()
        C = dict()
        for item,users in item_user.items():
            for u in users:
                N[u] = N.get(u,0)+1
                for v in users:
                    # print(u,v)
                    if u == v:
                        continue
                    C.setdefault(u,{})
                    C[u].setdefault(v,0)
                    # Penalize the top items
                    C[u][v] += 1/math.log(1+len(users),2)
        # calculate final similarity
        for u,related_users in C.items():
            self.usersim.setdefault(u,{})
            for v,cuv in related_users.items():
                self.usersim[u].setdefault(v,0)
                self.usersim[u][v] = cuv/math.sqrt(N[u]*N[v])
        '''

        usercnt = 0
        for user1 in range(12000):  # MemoryError without limits on user1 & user2
            self.usersim.setdefault(user1,{})
            for user2 in range(12000):
                if (user1 == user2): continue
                dot = 0  
                for item in self.train[user1]: 
                    if item in self.train[user2]:
                        dot += (self.train[user1][item] - self.aver_rate[user1]) * (self.train[user2][item] - self.aver_rate[user2]) 
                self.usersim[user1].setdefault(user2,0)
                if self.div[user1] != 0 and self.div[user2] != 0:   # else usersim[user1][user2] = 0
                    self.usersim[user1][user2] = dot / (self.div[user1] * self.div[user2])
            # print the progress
            usercnt += 1
            if usercnt % 1000 == 0:
                print(usercnt)
        
        for user1 in range(12000, 23599):
            self.usersim.setdefault(user1,{})
            for user2 in range(12000, 23599):
                if (user1 == user2): continue
                dot = 0  
                for item in self.train[user1]: 
                    if item in self.train[user2]:
                        dot += (self.train[user1][item] - self.aver_rate[user1]) * (self.train[user2][item] - self.aver_rate[user2]) 
                self.usersim[user1].setdefault(user2,0)
                if self.div[user1] != 0 and self.div[user2] != 0:   # else usersim[user1][user2] = 0
                    self.usersim[user1][user2] = dot / (self.div[user1] * self.div[user2])
            # print the progress
            usercnt += 1
            if usercnt % 1000 == 0:
                print(usercnt)


    def recommend(self, user, k=8, nitems=10):
        """Overwrite the function.
        :param user: the user we recommend items to
        :param train: dataset
        :param k: most similar k users
        :param nitem: most similar n items
        :return: ranks for top k items
        """
        rank = dict()
        simsum = dict()
        interacteditems = self.train[user]

        for user2, sim in sorted(self.usersim[user].items(),key = lambda x:x[1],reverse=True)[0:k]:
            for item, rate in self.train[user2].items():
                if (item in interacteditems) or (rate == -1): continue
                if item not in rank:
                    rank[item] = sim * (rate - self.aver_rate[user2])
                else:
                    rank[item] += sim * (rate - self.aver_rate[user2])
                if item not in simsum:
                    simsum[item] = sim
                else:
                    simsum[item] += sim

        for item in range(21602):
            if item not in rank: continue
            if simsum[item] != 0:
                rank[item] /= simsum[item]
        return dict(sorted(rank.items(),key = lambda x:x[1],reverse = True)[0:nitems])

    def solv(self, k=8, nitems=10):
        '''
        :param k:top k similar users
        :param nitems:top n similar items
        :return: recall,precision
        '''
        # hit = 0
        # recall = 0
        # precision = 0
        print("Recommending the first five items...")
        with open(Path(__file__).parent.parent.parent.joinpath('submit/best_result_UserCF.txt'),'w') as fw:
            begin_output = 1
            for user in self.train.keys():
                if begin_output == 1:
                    begin_output = 0
                else:
                    fw.write("\n")
                fw.write(str(user) + "\t")
                # tu = self.test.get(user, {})
                rank = self.recommend(user, k, nitems)
                flag = 0
                for item in rank:
                    if flag == 0:
                        fw.write(str(item))
                        flag = 1
                    else:
                        fw.write("," + str(item))
                if user % 1000 == 0:
                    print(user)
                '''
                for item,pui in rank.items():
                    if item in tu:
                        hit += 1
                recall += len(tu)
                precision += nitems
                print(tu,rank,recall,precision)
                '''
        # return hit/(recall*1.0) , hit/(precision*1.0)

    def coverage(self, k=8, nitems=10):
        '''
        :return: coverage of recommended items
        '''
        all_items = set()
        recommended_items = set()
        for user in self.train.keys():
            for item in self.train[user].keys():
                all_items.add(item)
            rank = self.recommend(user, k=k, nitems=nitems)
            for item,pvi in rank.items():
                recommended_items.add(item)
        return len(recommended_items)/(len(all_items) * 1.0)

    def popularity(self, k=8, nitems=10):
        """
        :return: average popularity of recommended items
        """
        item_popularity = dict()
        for user in self.train.keys():
            for item in self.train[user].keys():
                if item not in item_popularity:
                    item_popularity[item] = 0
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in self.test.keys():
            rank = self.recommend(user, k=k, nitems=nitems)
            for item,pui in rank.items():
                ret += math.log(1+item_popularity[item])
                n += 1
        ret /= n * 1.0
        return ret


if __name__ == '__main__':        
    uc = userCF()
    uc.initData()
    # print(len(uc.train.keys()))
    # print(uc.test)
    uc.userSimilarity_3(uc.train)
    k = 50        # k = 50 (result3), 500 (result1), 5000 (result2)
    nitems = 100
    uc.solv(k=k, nitems=nitems)
    # coverage = uc.coverage(k=k, nitems=nitems)
    # popularity = uc.popularity(k=k, nitems=nitems)