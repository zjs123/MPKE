import torch
import random as rd
from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, numOfTriple):
        self.tripleList = torch.LongTensor(range(numOfTriple))
        self.numOfTriple = numOfTriple

    def __len__(self):
        return self.numOfTriple

    def __getitem__(self, item):
        return self.tripleList[item]


class generateBatches:

    def __init__(self, batch, train2id, step_list, positiveBatch, corruptedBatch,numOfEntity, numOfRelation, headRelation2Tail, tailRelation2Head,headTail2Relation):
        self.batch = batch
        self.train2id = train2id
        self.step_list = step_list
        self.positiveBatch = positiveBatch
        self.corruptedBatch = corruptedBatch
        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head
        self.headTail2Relation = headTail2Relation

        self.generatePosAndCorBatch()
        rd.seed(0.5)

    def generatePosAndCorBatch(self):
        self.positiveBatch["h"] = []
        self.positiveBatch["r"] = []
        self.positiveBatch["t"] = []
        self.positiveBatch["time"] = []
        self.positiveBatch["step"] = []
        self.corruptedBatch["h"] = []
        self.corruptedBatch["r"] = []
        self.corruptedBatch["t"] = []
        self.corruptedBatch["time"] = []
        self.corruptedBatch["step"] = []
        for tripleId in self.batch:
            tmpHead = self.train2id["h"][tripleId]
            tmpRelation = self.train2id["r"][tripleId]
            tmpTail = self.train2id["t"][tripleId]
            tmpTime = self.train2id["time"][tripleId]
            tmpStep = self.train2id["step"][tripleId]
            #random=rd.random()
            for i in range(10):
                self.positiveBatch["h"].append(tmpHead)
                self.positiveBatch["r"].append(tmpRelation)
                self.positiveBatch["t"].append(tmpTail)
                self.positiveBatch["time"].append(tmpTime)
                self.positiveBatch["step"].append(tmpStep)
                random=torch.rand(1).item()
                if random <=0.5:
                    tmpCorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    while tmpCorruptedHead in self.tailRelation2Head[tmpTime][tmpTail][tmpRelation] or tmpCorruptedHead == tmpHead:
                        tmpCorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    self.corruptedBatch["h"].append(tmpCorruptedHead)
                    self.corruptedBatch["r"].append(tmpRelation)
                    self.corruptedBatch["t"].append(tmpTail)
                    self.corruptedBatch["time"].append(tmpTime)

                
                elif random<=1:
                    tmpCorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    #print(tmpCorruptedTail)
                    while tmpCorruptedTail in self.headRelation2Tail[tmpTime][tmpHead][tmpRelation] or tmpCorruptedTail == tmpTail:
                        tmpCorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    self.corruptedBatch["h"].append(tmpHead)
                    self.corruptedBatch["r"].append(tmpRelation)
                    self.corruptedBatch["t"].append(tmpCorruptedTail)
                    self.corruptedBatch["time"].append(tmpTime)
                
                else:
                    tmpCorruptedRel = torch.FloatTensor(1).uniform_(0, self.numOfRelation).long().item()
                    #print(tmpCorruptedTail)
                    while tmpCorruptedRel in self.headTail2Relation[tmpTime][tmpHead][tmpTail] or tmpCorruptedRel == tmpRelation:
                        tmpCorruptedRel = torch.FloatTensor(1).uniform_(0, self.numOfRelation).long().item()
                    self.corruptedBatch["h"].append(tmpHead)
                    self.corruptedBatch["r"].append(tmpCorruptedRel)
                    self.corruptedBatch["t"].append(tmpTail)
                    self.corruptedBatch["time"].append(tmpTime)
                
                if len(self.step_list[tmpHead])==1:
                    rand_step = 1
                else:
                    rand_step=rd.randint(0,len(self.step_list[tmpHead])-1)
                    while rand_step == tmpStep:
                        rand_step=rd.randint(0,len(self.step_list[tmpHead])-1)
                self.corruptedBatch["step"].append(rand_step)
        for aKey in self.positiveBatch:
            self.positiveBatch[aKey] = torch.LongTensor(self.positiveBatch[aKey])
        for aKey in self.corruptedBatch:
            self.corruptedBatch[aKey] = torch.LongTensor(self.corruptedBatch[aKey])

