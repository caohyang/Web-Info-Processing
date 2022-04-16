### PLATFORM: AI Studio
import random
import time
import os
import math
import codecs
import copy
import numpy as np
from gensim.models import word2vec

# TransE ONLY
entity2id = {}
relation2id = {}
entityId2vec = {}
relationId2vec = {}
loss_ls = []

def data_loader(file, status):
    if status == 0:
        file1 = file + "train.txt"       
    elif status == 1:
        file1 = file + "dev.txt"
    else:
        file1 = file + "test.txt"
    file2 = file + "entity_with_text.txt"
    file3 = file + "relation_with_text.txt"
    entity_set = set()
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = triple[0]
            if status != 2:
                t_ = triple[2]          # be careful with the index
            else:
                t_ = -1
            r_ = triple[1]
            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)
            relation_set.add(r_)

    with codecs.open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            entity_id = line.strip().split("\t")[0]
            entity_set.add(entity_id)

    with codecs.open(file3, 'r') as f:
        content = f.readlines()
        for line in content:
            relation_id = line.strip().split("\t")[0]
            relation_set.add(relation_id)

    return entity_set, relation_set, triple_list

def distanceL2(h, r, t):
    # 为方便求梯度，去掉sqrt
    return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
    return np.sum(np.fabs(h + r - t))

def distance(h, r, t):
    h = np.array(h)
    r = np.array(r)
    t = np.array(t)
    s = h + r - t
    return np.linalg.norm(s)

def transE_loader(file):
    file1 = file + "entity_50dim_batch400"
    file2 = file + "relation_50dim_batch400"
    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split("\t")
            entityId2vec[line[0]] = eval(line[1])
    with codecs.open(file2, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split("\t")
            relationId2vec[line[0]] = eval(line[1])

class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1            # self.L1 = False -> use L2 distance
        self.loss = 0

    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        self.relation = relation_dict
        self.entity = entity_dict

    def train(self, epochs):
        nbatches = 400
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            # Sbatch: list
            Sbatch = random.sample(self.triple_list, batch_size)
            Tbatch = []

            for triple in Sbatch:
                corrupted_triple = self.Corrupt(triple)
                if (triple, corrupted_triple) not in Tbatch:
                    Tbatch.append((triple, corrupted_triple))
            self.update_embeddings(Tbatch)

            end = time.time()
            if epoch % 100 == 0:
                print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
                print("loss: ", self.loss)
                loss_ls.append(self.loss)
                with codecs.open("entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(r + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")

        print("Writing model file...")
        with codecs.open("entity_50dim_batch400", "w") as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("relation_50dim_batch400", "w") as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")

        with codecs.open("loss", "w") as f3:
            f3.write(str(loss_ls))

        print("Write completed.")

    def Corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random()
        if seed > 0.5:
            # 替换head
            rand_head = triple[0]
            while rand_head == triple[0]:
                rand_head = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[0] = rand_head

        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:
            # 取copy里的vector累积更新
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]
            relation_update = copy_relation[triple[2]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]

            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err

                grad_pos = 2 * (h_correct + relation - t_correct)
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                # 梯度计算
                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                # 梯度下降
                # head系数为正，减梯度；tail系数为负，加梯度
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                # corrupt项整体为负，因此符号与correct相反
                if triple[0] == corrupted_triple[0]:  # 若替换的是尾实体，则头实体更新两次
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg

                elif triple[1] == corrupted_triple[1]:  # 若替换的是头实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                # relation更新两次
                relation_update -= self.learning_rate * grad_pos
                relation_update -= (-1) * self.learning_rate * grad_neg

        # batch norm
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        # 达到批量更新的目的
        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)

def eva(entity_set, triple_list, test=False):
    # triple_batch = random.sample(triple_list, 100)
    triple_batch = triple_list                                                                                                                                                                                                          
    hit1 = 0
    hit5 = 0
    testcnt = 0
    file = "/home/aistudio/test_output.txt"             # you may need to reset the path
    f = codecs.open(file, "w")
    
    for triple in triple_batch:
        if test == True and testcnt > 0:
            f.write("\n")
        testcnt += 1 
        dlist = []
        h = triple[0]
        r = triple[2]
        
        if test == False:
            t = triple[1]
            dlist.append((t, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t])))
        else: 
            if h not in entityId2vec:
                f.write("1,2,3,4,5")
                continue

        for t_ in entity_set:
            if t_ not in entityId2vec:
                continue
            dlist.append((t_, distance(entityId2vec[h], relationId2vec[r], entityId2vec[t_])))
        dlist = sorted(dlist, key=lambda val: val[1])

        if test == True:
            for index in range(5):
                if index != 4:
                    f.write(str(dlist[index][0]) + ",")
                else:
                    f.write(str(dlist[index][0]))
        else:
            for index in range(5):
                if dlist[index][0] == t:
                    if index < 1:
                        hit1 += 1
                    if index < 5:
                        hit5 += 1
                    break
        if testcnt % 500 == 0:
            print(testcnt)
    if test == False:
        print("hit@1:", hit1 / len(triple_batch))
        print("hit@5:", hit5 / len(triple_batch))

if __name__ == '__main__':
    file1 = "/home/aistudio/"               # you may need to reset the path
    entity_set, relation_set, triple_list = data_loader(file1, status = 0)   # train: test=0, dev: test=1, test: test=2
    print("Load file...")
    print("Load completed. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs = 2000)

    # dev
    print("Load dev file...")
    entity_set, relation_set, triple_list = data_loader(file1, status = 1)  
    print("Load completed. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    print("Load transE vec...")
    transE_loader("/home/aistudio/src/method1/")
    print("Load completed. Evaluating dev file...")
    eva(entity_set, triple_list, test=False)

    # test
    print("Load test file...")
    entity_set, relation_set, triple_list = data_loader(file1, status = 2)  
    print("Load completed. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    print("Load transE vec...")
    transE_loader("/home/aistudio/src/method1/")
    print("Load completed. Evaluating test file...")
    eva(entity_set, triple_list, test=True)