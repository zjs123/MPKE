# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:04:27 2020

@author: zjs
"""

import re
import pickle
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MPKE, MPKE_Trans
from readTrainingData import readData
from generatePosAndCorBatch import generateBatches, dataset
import random
import Loss
import numpy as np

class Train:
    
    def __init__(self,args):
        self.entityDimension = args.dimension
        self.relationDimension = args.dimension
        self.numOfEpochs = args.numOfEpochs
        self.numOfBatches = args.numOfBatches
        self.learningRate = args.lr
        self.margin_triple = args.margin
        self.margin_relation = args.margin
        self.norm = args.norm
        self.dataset = args.dataset
        self.norm_m = args.norm_m
        self.hyper_m = args.hyper_m
        self.ns = args.ns
        
        self.Triples = None
        self.train2id = {}
        self.year2id = {}
        self.step_list = {}
        self.headRelation2Tail = {}
        self.tailRelation2Head = {}
        self.headTail2Relation = {}
        self.positiveBatch = {}
        self.corruptedBatch = {}
        self.relation_pair_batch = {}
        
        self.nums = [0,0,0,0,0]
        self.numOfTrainTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        self.numOfTime = 0
        self.numOfMaxLen = 0
        
        self.validate2id = {}
        self.validateHead = None
        self.validateRelation = None
        self.validateTail = None
        self.validateTime = None
        self.numOfValidateTriple = 0

        self.test2id = {}
        self.testHead = None
        self.testRelation = None
        self.testTail = None
        self.testTime = None
        self.numOfTestTriple = 0
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.setup_seed(1)
        
        self.model = None

        self.train()#train the model
        
        self.write()#write the result and save model
        
        self.test()#run the test dataset
    
    
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def train(self):
        path = "./dataset/"+self.dataset
        data=readData(path, self.train2id, self.year2id,self.step_list,self.headRelation2Tail,self.tailRelation2Head,
                        self.headTail2Relation,self.nums)#read training data
        
        self.Triples = data.out()
        
        self.numOfTrainTriple = self.nums[0]
        self.numOfEntity = self.nums[1]
        self.numOfRelation = self.nums[2]
        self.numOfTime = self.nums[3]
        self.numOfMaxLen = self.nums[4]
        
        self.readValidateTriples(path)
        self.readTestTriples(path)
        
        
        self.model = MPKE(self.numOfEntity, self.numOfRelation, self.numOfTime, self.numOfMaxLen, self.entityDimension, self.relationDimension,
                      self.norm,self.norm_m,self.hyper_m)#init the model

        #self.preRead()


        self.model.to(self.device)
        
        #self.test()

        Margin_Loss_D = Loss.double_marginLoss()
        #Margin_Loss_H = Loss.marginLoss()
        #Margin_Loss_S = Loss.sigmoidLoss()
        
        optimizer = optim.Adam(self.model.parameters(),lr=self.learningRate)
        
        Dataset = dataset(self.numOfTrainTriple)
        batchsize = int(self.numOfTrainTriple / self.numOfBatches)
        dataLoader = DataLoader(Dataset, batchsize, True)

        Log_path = "./dataset/"+self.dataset + "/"+str(self.learningRate)+"_MP_"+"log.txt"
        Log = open(Log_path, "w")
        for epoch in range(self.numOfEpochs):
            epochLoss = 0
            for batch in dataLoader:
                self.positiveBatch = {}
                self.corruptedBatch = {}
                generateBatches(batch, self.train2id, self.step_list, self.positiveBatch, self.corruptedBatch,self.numOfEntity,
                                self.numOfRelation,self.headRelation2Tail, self.tailRelation2Head, self.headTail2Relation, self.ns)
                optimizer.zero_grad()
                positiveBatchHead = self.positiveBatch["h"].to(self.device)
                positiveBatchRelation = self.positiveBatch["r"].to(self.device)
                positiveBatchTail = self.positiveBatch["t"].to(self.device)
                positiveBatchTime = self.positiveBatch["time"].to(self.device)
                positiveBatchStep = self.positiveBatch["step"].to(self.device)
                corruptedBatchHead = self.corruptedBatch["h"].to(self.device)
                corruptedBatchRelation = self.corruptedBatch["r"].to(self.device)
                corruptedBatchTail = self.corruptedBatch["t"].to(self.device)
                corruptedBatchTime = self.corruptedBatch["time"].to(self.device)
                corruptedBatchStep = self.corruptedBatch["step"].to(self.device)
                
                positiveScore, negativeScore = self.model(positiveBatchHead, positiveBatchRelation, positiveBatchTail,positiveBatchTime,positiveBatchStep,corruptedBatchHead,
                                   corruptedBatchRelation, corruptedBatchTail,corruptedBatchTime,corruptedBatchStep)

                ent_embeddings = self.model.entity_embeddings(torch.cat([positiveBatchHead, positiveBatchTail, corruptedBatchHead, corruptedBatchTail]))
                rel_embeddings = self.model.relation_embeddings(torch.cat([positiveBatchRelation,corruptedBatchRelation]))

                loss = Margin_Loss_D(positiveScore, negativeScore, self.margin_triple)
                
                time_embeddings = self.model.time_embeddings(positiveBatchTime)
                step_embeddings = self.model.step_embeddings(positiveBatchStep)

                batchLoss = loss+Loss.normLoss(time_embeddings)+Loss.normLoss(step_embeddings)
                batchLoss.backward()
                optimizer.step()
                epochLoss += batchLoss
               
            print ("epoch " + str(epoch) + ": , loss: " + str(epochLoss))

            if epoch%20 ==0 and epoch!=0:
                Log.write("epoch " + str(epoch) + ": , loss: " + str(epochLoss))
                meanRank_H,Hits10_H= self.model.Validate_entity_H(validateHead=self.testHead.to(self.device), validateRelation=self.testRelation.to(self.device), validateTail=self.testTail.to(self.device),validateTime=self.testTime.to(self.device),validateStepH=self.testStepH.to(self.device),trainTriple=self.Triples.to(self.device),numOfvalidateTriple=self.numOfTestTriple)
                print("mean rank H_2_2_.1_nonorm: "+str(meanRank_H))
                meanRank_T,Hits10_T= self.model.Validate_entity_T(validateHead=self.testHead.to(self.device), validateRelation=self.testRelation.to(self.device), validateTail=self.testTail.to(self.device),validateTime=self.testTime.to(self.device),validateStepT=self.testStepT.to(self.device),trainTriple=self.Triples.to(self.device),numOfvalidateTriple=self.numOfTestTriple)
                print("mean rank T_2_2_.1_nonorm: "+str(meanRank_T))
                Log.write("valid H MR: "+str(meanRank_H)+"\n")
                Log.write("valid T MR: "+str(meanRank_T)+"\n")
                Log.write("valid entity MR: "+str((meanRank_H+meanRank_T)/2)+"\n")
                print("valid entity MR: "+str((meanRank_H+meanRank_T)/2))
                Log.write("valid entity H10: "+str((Hits10_H+Hits10_T)/2)+"\n")
                print("valid entity H10: "+str((Hits10_H+Hits10_T)/2))
                ValidMR_relation = self.model.fastValidate_relation(validateHead=self.testHead.to(self.device), validateRelation=self.testRelation.to(self.device), validateTail=self.testTail.to(self.device),validateTime=self.testTime.to(self.device),validateStepH=self.testStepH.to(self.device),numOfvalidateTriple=self.numOfTestTriple)
                Log.write("valid relation MR: "+str(ValidMR_relation)+"\n")
                Log.write("\n")
                print("valid relation MR: "+str(ValidMR_relation))
        Log.close()
            
    
    def test(self):
        
        meanRank_r, Hits1_r, meanRank_rF, Hits1_rF,MRR_r,MRR_rF= self.model.test_relation(testHead=self.testHead.to(self.device), testRelation=self.testRelation.to(self.device), testTail=self.testTail.to(self.device),testTime=self.testTime.to(self.device),testStepH=self.testStepH.to(self.device),trainTriple=self.Triples.to(self.device),numOfTestTriple=self.numOfTestTriple)
        print(torch.sum(torch.abs(self.model.mod_r_embeddings.weight.data),1))
        print("test_mean_rank_R: "+str(meanRank_r))
        print("test_Hits_1_R: "+str(Hits1_r))
        print("test_mean_rank_FR: "+str(meanRank_rF))
        print("test_Hits_1_FR: "+str(Hits1_rF))
        
        meanRank_H, Hits10_H, meanRank_F_H, Hits10_F_H, Hits3_H, Hits1_H,Hits3_HF,Hits1_HF,MRR_H,MRR_HF = self.model.test_entity_H(testHead=self.testHead.to(self.device), testRelation=self.testRelation.to(self.device), testTail=self.testTail.to(self.device),testTime=self.testTime.to(self.device),testStepH=self.testStepH.to(self.device),trainTriple=self.Triples.to(self.device),numOfTestTriple=self.numOfTestTriple)
        
        meanRank_T, Hits10_T, meanRank_F_T, Hits10_F_T, Hits3_T, Hits1_T,Hits3_TF,Hits1_TF ,MRR_T,MRR_TF= self.model.test_entity_T(testHead=self.testHead.to(self.device), testRelation=self.testRelation.to(self.device), testTail=self.testTail.to(self.device),testTime=self.testTime.to(self.device),testStepT=self.testStepT.to(self.device),trainTriple=self.Triples.to(self.device),numOfTestTriple=self.numOfTestTriple)
        meanRank = (meanRank_H+meanRank_T)/2
        Hits10 = (Hits10_H+Hits10_T)/2
        meanRank_F = (meanRank_F_H+meanRank_F_T)/2
        Hits10_F = (Hits10_F_H+Hits10_F_T)/2
        Hits3 = (Hits3_H+Hits3_T)/2
        Hits1 = (Hits1_H+Hits1_T)/2
        Hits3_F = (Hits3_HF+Hits3_TF)/2
        Hits1_F = (Hits1_HF+Hits1_TF)/2
        MRR = (MRR_H+MRR_T)/2
        MRR_F = (MRR_HF+MRR_TF)/2
        print("test_mean_rank_E: "+str(meanRank))
        print("test_Hits_10_E: "+str(Hits10))
        print("test_mean_rank_FE: "+str(meanRank_F))
        print("test_Hits_10_FE: "+str(Hits10_F))
        print("Hits3: "+str(Hits3))
        print("Hits1: "+str(Hits1))

        self.print_result(meanRank, meanRank_F, Hits10, Hits10_F,Hits3, Hits1, Hits3_F,Hits1_F,MRR,MRR_F,meanRank_r, meanRank_rF, Hits1_r, Hits1_rF,MRR_r,MRR_rF)


    def write(self):
        #print "-----Writing Training Results to " + self.outAdd + "-----"
        model_path = "./dataset/"+self.dataset + "/model.pickle"
        modelOutput = open(model_path, "wb")
        pickle.dump(self.model, modelOutput)
        modelOutput.close()

    def print_result(self,mr_e,mr_ef,h10,h10_f,h3,h1,h3_f,h1_f,mrr,mrr_f,mr_r,mr_rf,h1r,h1f,mrr_r,mrr_rf):
        model_path = "./dataset/"+self.dataset + "/"+str(self.learningRate)+"_MP_"+"Results.txt"
        Output = open(model_path, "w")
        Output.write("entity MR | MRF: "+str(mr_e)+" | "+str(mr_ef)+"\n")
        Output.write("entity MRR | MRRF: "+str(mrr)+" | "+str(mrr_f)+"\n")
        Output.write("entity Hits_RAW 1|3|10: "+str(h1)+" | "+str(h3)+" | "+str(h10)+"\n")
        Output.write("entity Hits_F 1|3|10: "+str(h1_f)+" | "+str(h3_f)+" | "+str(h10_f)+"\n")
        Output.write("relation MR | MRF: "+str(mr_r)+" | "+str(mr_rf)+"\n")
        Output.write("relation MRR | MRRF: "+str(mrr_r)+" | "+str(mrr_rf)+"\n")
        Output.write("relation Hits1|1f: "+str(h1r)+" | "+str(h1f)+"\n")

        Output.close()

    def preRead(self):
        #print "-----Reading Pre-Trained Results from " + self.preAdd + "-----"
        modelInput = open("./dataset/"+self.dataset + "/model.pickle", "rb")
        self.model = pickle.load(modelInput)
        modelInput.close()
        

    def readValidateTriples(self,path):
        fileName = path+"/valid.txt"
        print ("-----Reading Validation Triples from " +fileName + "-----")
        count = 0
        self.validate2id["h"] = []
        self.validate2id["r"] = []
        self.validate2id["t"] = []
        self.validate2id["time"] = []
        self.validate2id["step"] = []
        inputData = open(fileName)
        line = inputData.readline()
        self.numOfValidateTriple = int(re.findall(r"\d+", line)[0])
        lines = inputData.readlines()
        count=0
        for line in lines:
            count=count+1
            head=int(line.strip().split()[0])
            relation=int(line.strip().split()[1])
            tail=int(line.strip().split()[2])
            time = line.strip().split()[3]
            year = time.strip().split('-')[0]
            '''
            month = time.strip().split('-')[1]
            day = time.strip().split('-')[2]
            year = month*100+day
            '''
            tmpTime = ""
            if '#' not in year:
                year = int(year)
            else:
                #year = int(year.replace("#","0"))
                year=3000
            if year not in self.year2id:
                tmpTime = 0
            else:
                tmpTime = self.year2id[year]
            '''
            for key in self.year2id.keys():
                if year <= key-1:
                    tmpTime = self.year2id[key]-1
                    break
            if type(tmpTime) != int:
                tmpTime = self.year2id[list(self.year2id.keys())[-1]]
            '''
            step = 0
            if head not in self.step_list.keys():
                step=0
            else:
                for time in self.step_list[head]:
                    if time < year:
                        step+=1
                    else:
                        break
            self.validate2id["h"].append(head)
            self.validate2id["r"].append(relation)
            self.validate2id["t"].append(tail)
            self.validate2id["time"].append(tmpTime)
            self.validate2id["step"].append(step)
        inputData.close()
        #print(self.validate2id["step"])
        if count == self.numOfValidateTriple:
            print ("number of validation triples: " + str(self.numOfValidateTriple))
            self.validateHead = torch.LongTensor(self.validate2id["h"])
            self.validateRelation = torch.LongTensor(self.validate2id["r"])
            self.validateTail = torch.LongTensor(self.validate2id["t"])
            self.validateTime = torch.LongTensor(self.validate2id["time"])
            self.validateStep = torch.LongTensor(self.validate2id["step"])
        else:
            print ("error in " + fileName)
            
    def readTestTriples(self,path):
        fileName =path+"/test.txt"
        print ("-----Reading Test Triples from " + fileName + "-----")
        count = 0
        self.test2id["h"] = []
        self.test2id["r"] = []
        self.test2id["t"] = []
        self.test2id["time"] = []
        self.test2id["step_H"] = []
        self.test2id["step_T"] = []
        inputData = open( fileName)
        line = inputData.readline()
        self.numOfTestTriple = int(re.findall(r"\d+", line)[0])
        lines = inputData.readlines()
        count=0
        for line in lines:
            count=count+1
            head=int(line.strip().split()[0])
            relation=int(line.strip().split()[1])
            tail=int(line.strip().split()[2])
            time = line.strip().split()[3]
            year = time.strip().split('-')[0]
            '''
            month = time.strip().split('-')[1]
            day = time.strip().split('-')[2]
            year = month*100+day
            '''
            tmpTime = ""
            if '#' not in year:
                year = int(year)
            else:
                #year = int(year.replace("#","0"))
                year=3000
            if year not in self.year2id:
                tmpTime = 0
            else:
                tmpTime = self.year2id[year]
            '''
            for key in self.year2id.keys():
                if year <= key-1:
                    tmpTime = self.year2id[key]-1
                    break
            if type(tmpTime) != int:
                tmpTime = self.year2id[list(self.year2id.keys())[-1]]
            '''
            stepH = 0
            if head not in self.step_list.keys():
                stepH=0
            else:
                for time in self.step_list[head]:
                    if time < year:
                        stepH+=1
                    else:
                        break
            stepT = 0
            if tail not in self.step_list.keys():
                stepT=0
            else:
                for time in self.step_list[tail]:
                    if time < year:
                        stepT+=1
                    else:
                        break
            self.test2id["h"].append(head)
            self.test2id["r"].append(relation)
            self.test2id["t"].append(tail)
            self.test2id["time"].append(tmpTime)
            self.test2id["step_H"].append(stepH)
            self.test2id["step_T"].append(stepT)
        inputData.close()
        #print(self.test2id["step_H"])
        if count == self.numOfTestTriple:
            print ("number of test triples: " + str(self.numOfTestTriple))
            self.testHead = torch.LongTensor(self.test2id["h"])
            self.testRelation = torch.LongTensor(self.test2id["r"])
            self.testTail = torch.LongTensor(self.test2id["t"])
            self.testTime=torch.LongTensor(self.test2id["time"])
            self.testStepH=torch.LongTensor(self.test2id["step_H"])
            self.testStepT=torch.LongTensor(self.test2id["step_T"])
        else:
            print ("error in " + fileName)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model")
    parser.add_argument("--hidden",dest="dimension",type=int,default=128)
    parser.add_argument("--nbatch",dest="numOfBatches",type=int,default=100)
    parser.add_argument("--nepoch",dest="numOfEpochs",type=int,default=500)
    parser.add_argument("--margin",dest="margin",type=float,default=12)
    parser.add_argument("--dataset",dest="dataset",type=str,default="YAGO11K")
    #[YAGO11K,WIKI12K,WIKI11K]
    parser.add_argument("--lr",dest="lr",type=float,default=0.005)#learning rate
    parser.add_argument("--norm",dest="norm",type=int,default=1)#the norm of semantic part
    parser.add_argument("--norm_m",dest="norm_m",type=int,default=2)#the norm of structure part
    parser.add_argument("--hyper_m",dest="hyper_m",type=float,default=0.5)#trade off to balance two parts of our model
    parser.add_argument("--ns",dest="ns",type=int,default=10)#negative sampling ratio
    
    args=parser.parse_args()
    Train(args)