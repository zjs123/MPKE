import re
import os
import torch
import pickle
import copy
from collections import defaultdict as ddict

class readData:
    def __init__(self, inAdd, train2id, year2id, step_list, headRelation2Tail, tailRelation2Head, headTail2Relation,nums):
        self.inAdd = inAdd
        self.train2id = train2id
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head
        self.headTail2Relation= headTail2Relation
        self.nums = nums
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        self.numOfTime = 0
        self.numOfMaxLen = 0
        self.year2id = year2id
        self.step_list = step_list
        
        self.trainTriple = None

        self.readTriple2id()

        self.readTrain2id()
        print ("number of triples: " + str(self.numOfTrainTriple))
        print ("num Of time: "+str(self.numOfTime))
        print ("num of max_step: "+str(self.numOfMaxLen))

        self.readEntity2id()
        print ("number of entities: " + str(self.numOfEntity))

        self.readRelation2id()
        print ("number of relations: " + str(self.numOfRelation))


        self.nums[0] = self.numOfTrainTriple
        self.nums[1] = self.numOfEntity
        self.nums[2] = self.numOfRelation
        self.nums[3] = self.numOfTime
        self.nums[4] = self.numOfMaxLen

    
    def out(self):
        return self.trainTriple  
    

    def readTriple2id(self):
        years={}
        step_set={}
        print ("-----Reading triple2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/train.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfTriple = int(re.findall(r"\d+", line)[0])
        print("num of triple= "+str(self.numOfTriple))
        #self.train2id["time"] = []
        self.trainTriple = torch.ones(self.numOfTriple, 4)
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(line.strip().split()[0])
                #tmpTail = int(re.findall(r"\d+", line)[1])
                #tmpRelation = int(re.findall(r"\d+", line)[2])
                tmpTail = int(line.strip().split()[2])
                tmpRelation = int(line.strip().split()[1])
                time = line.strip().split()[3]
                year = time.strip().split('-')[0]
                '''
                month = time.strip().split('-')[1]
                day = time.strip().split('-')[2]
                year = month*100+day
                '''
                #print(time)
                if '#' not in year:
                    year = int(year)
                else:
                    year=3000
                    #year = int(year.replace("#","0"))
                if year not in years.keys():
                    years[year]=1
                else:
                    years[year]+=1
                if tmpHead not in step_set.keys():
                    step_set[tmpHead] = set()
                step_set[tmpHead].add(year)
                count += 1
                line = inputData.readline()
            else:
                print ("error in triple2id.txt at Line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        year_list = sorted(years.keys())
        index=1
        T_count = 0
        self.numOfMaxLen = 0
        self.year2id[0]=0
        
        for i in year_list:
            self.year2id[i] = index
            index=index+1
        '''
        pre = year_list[0]
        for i in year_list:
            T_count+=years[i]
            if T_count >=500:
                self.year2id[pre+1] = index
                index+=1
                pre = i
                T_count=0
        '''

        self.numOfTime = len(self.year2id)+1
        for key in step_set.keys():
            if self.numOfMaxLen < len(step_set[key]):
                self.numOfMaxLen = len(step_set[key])
            self.step_list[key] = list(step_set[key])
        if count == self.numOfTriple:
            self.trainTriple.long()
            return
        else:
            print ("error in triple2id.txt  count= "+str(count))
            return

    def readTrain2id(self):
        print ("-----Reading train2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/train.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfTrainTriple = int(re.findall(r"\d+", line)[0])
        self.train2id["h"] = []
        self.train2id["t"] = []
        self.train2id["r"] = []
        self.train2id["time"] = []
        self.train2id["step"] = []
        self.trainTriple = torch.ones(self.numOfTriple, 4)
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                #tmpTail = int(re.findall(r"\d+", line)[1])
                #tmpRelation = int(re.findall(r"\d+", line)[2])
                tmpTail = int(re.findall(r"\d+", line)[2])
                tmpRelation = int(re.findall(r"\d+", line)[1])
                time = line.strip().split()[3]
                year = time.strip().split('-')[0]
                '''
                month = time.strip().split('-')[1]
                day = time.strip().split('-')[2]
                year = month*100+day
                '''
                tmpTime=""
                #print(year)
                if '#' not in year:
                    year = int(year)
                else:
                    #year = int(year.replace("#","0"))
                    year=3000
                #print(year)
                tmpTime = self.year2id[year]
                step = 0
                for time in self.step_list[tmpHead]:
                    if time < year:
                        step+=1
                    else:
                        break
                
                if tmpTime not in self.headRelation2Tail.keys():
                    self.headRelation2Tail[tmpTime] = {}
                    self.headRelation2Tail[tmpTime][tmpHead] = {}
                    self.headRelation2Tail[tmpTime][tmpHead][tmpRelation] = []
                    self.headRelation2Tail[tmpTime][tmpHead][tmpRelation].append(tmpTail)
                else:
                    if tmpHead not in self.headRelation2Tail[tmpTime].keys():
                         self.headRelation2Tail[tmpTime][tmpHead] = {}
                         self.headRelation2Tail[tmpTime][tmpHead][tmpRelation] = []
                         self.headRelation2Tail[tmpTime][tmpHead][tmpRelation].append(tmpTail)
                    else:
                        if tmpRelation not in self.headRelation2Tail[tmpTime][tmpHead].keys():
                            self.headRelation2Tail[tmpTime][tmpHead][tmpRelation] = []
                            self.headRelation2Tail[tmpTime][tmpHead][tmpRelation].append(tmpTail)
                        elif tmpTail not in self.headRelation2Tail[tmpTime][tmpHead][tmpRelation]:
                            self.headRelation2Tail[tmpTime][tmpHead][tmpRelation].append(tmpTail)
                            
                            
                if tmpTime not in self.tailRelation2Head.keys():
                    self.tailRelation2Head[tmpTime] = {}
                    self.tailRelation2Head[tmpTime][tmpTail] = {}
                    self.tailRelation2Head[tmpTime][tmpTail][tmpRelation] = []
                    self.tailRelation2Head[tmpTime][tmpTail][tmpRelation].append(tmpHead)
                else:
                    if tmpTail not in self.tailRelation2Head[tmpTime].keys():
                        self.tailRelation2Head[tmpTime][tmpTail] = {}
                        self.tailRelation2Head[tmpTime][tmpTail][tmpRelation] = []
                        self.tailRelation2Head[tmpTime][tmpTail][tmpRelation].append(tmpHead)
                    else:
                        if tmpRelation not in self.tailRelation2Head[tmpTime][tmpTail].keys():
                            self.tailRelation2Head[tmpTime][tmpTail][tmpRelation] = []
                            self.tailRelation2Head[tmpTime][tmpTail][tmpRelation].append(tmpHead)
                        elif tmpHead not in  self.tailRelation2Head[tmpTime][tmpTail][tmpRelation]:
                            self.tailRelation2Head[tmpTime][tmpTail][tmpRelation].append(tmpHead)
                            
                            
                if tmpTime not in self.headTail2Relation.keys():
                    self.headTail2Relation[tmpTime] = {}
                    self.headTail2Relation[tmpTime][tmpHead] = {}
                    self.headTail2Relation[tmpTime][tmpHead][tmpTail] = []
                    self.headTail2Relation[tmpTime][tmpHead][tmpTail].append(tmpRelation)
                else:
                    if tmpHead not in self.headTail2Relation[tmpTime].keys():
                       self.headTail2Relation[tmpTime][tmpHead] = {}
                       self.headTail2Relation[tmpTime][tmpHead][tmpTail] = []
                       self.headTail2Relation[tmpTime][tmpHead][tmpTail].append(tmpRelation)
                    else:
                        if tmpTail not in self.headTail2Relation[tmpTime][tmpHead].keys():
                            self.headTail2Relation[tmpTime][tmpHead][tmpTail] = []
                            self.headTail2Relation[tmpTime][tmpHead][tmpTail].append(tmpRelation)
                        elif tmpRelation not in self.headTail2Relation[tmpTime][tmpHead][tmpTail]:
                            self.headTail2Relation[tmpTime][tmpHead][tmpTail].append(tmpRelation)
                
                #print(step)
                self.train2id["h"].append(tmpHead)
                self.train2id["t"].append(tmpTail)
                self.train2id["r"].append(tmpRelation)
                self.train2id["time"].append(tmpTime)
                #print(tmpTime)
                self.train2id["step"].append(step)
                #print(step)
                self.trainTriple[count, 0] = tmpHead
                self.trainTriple[count, 1] = tmpRelation
                self.trainTriple[count, 2] = tmpTail
                self.trainTriple[count, 3] = tmpTime
                count += 1
                line = inputData.readline()
            else:
                print ("error in train2id.txt at Line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfTrainTriple:
            return
        else:
            print ("error in train2id.txt")
            return


    def readEntity2id(self):
        print ("-----Reading entity2id.txt from " + self.inAdd + "/-----")
        inputData = open(self.inAdd + "/entity2id.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfEntity = int(re.findall(r"\d+", line)[0])
        return


    def readRelation2id(self):
        print ("-----Reading relation2id.txt from " + self.inAdd + "/-----")
        inputData = open(self.inAdd + "/relation2id.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfRelation = int(re.findall(r"\d+", line)[0])
        return
