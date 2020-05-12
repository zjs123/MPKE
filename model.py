# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:03:24 2020

@author: zjs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MPKE(nn.Module):
    
    def __init__(self, numOfEntity, numOfRelation, numOfTime, numOfMaxLen, entityDimension, relationDimension, norm):
        super(MPKE,self).__init__()

        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.numOfTime = numOfTime
        self.numOfMaxLen = numOfMaxLen
        self.entityDimension = entityDimension
        self.relationDimension = relationDimension
        self.norm = norm
        
        self.pi = 3.14159262358979323846

        self.relation_embeddings = nn.Embedding(self.numOfRelation, self.relationDimension)
        nn.init.xavier_normal_(
            tensor=self.relation_embeddings.weight.data
        )
        
        self.mod_r_embeddings = nn.Embedding(self.numOfRelation, self.relationDimension)
        nn.init.xavier_normal_(
            tensor=self.relation_embeddings.weight.data
        )
        self.entity_embeddings = nn.Embedding(self.numOfEntity, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.entity_embeddings.weight.data
        )
        
        self.mod_e_embeddings = nn.Embedding(self.numOfEntity, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.entity_embeddings.weight.data
        )
        
        self.time_embeddings = nn.Embedding(self.numOfTime, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.entity_embeddings.weight.data
        )
        
        self.step_embeddings = nn.Embedding(self.numOfMaxLen, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.step_embeddings.weight.data
        )

        self.hyper_p=1
        self.hyper_m=0.5#0.01
        self.norm_m=2
        

    def forward(self, positiveBatchHead, positiveBatchRelation, positiveBatchTail,positiveBatchTime, positiveBatchStep,corruptedBatchHead, corruptedBatchRelation, corruptedBatchTail, corruptedBatchTime, corruptedBatchStep):

        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        #print(corruptedBatchTime)
        pH_embeddings = self.entity_embeddings(positiveBatchHead)
        pR_embeddings = self.relation_embeddings(positiveBatchRelation)
        pT_embeddings = self.entity_embeddings(positiveBatchTail)
        pTime_embedding = self.time_embeddings(positiveBatchTime)
        #print(corruptedBatchStep)
        pStep_embedding = torch.index_select(step_embeddings,0,positiveBatchStep)

        nH_embeddings = self.entity_embeddings(corruptedBatchHead)
        nR_embeddings = self.relation_embeddings(corruptedBatchRelation)
        nT_embeddings = self.entity_embeddings(corruptedBatchTail)
        nTime_embedding = self.time_embeddings(corruptedBatchTime)
        #print(len(step_embeddings))
        nStep_embedding = torch.index_select(step_embeddings,0,corruptedBatchStep)

        H_mod_embeddings = torch.abs(self.mod_e_embeddings(positiveBatchHead))
        R_mod_embeddings = torch.abs(self.mod_r_embeddings(positiveBatchRelation))
        T_mod_embeddings = torch.abs(self.mod_e_embeddings(positiveBatchTail))

        pH_embeddings = pH_embeddings#*self.pi
        pR_embeddings = pR_embeddings*self.pi
        pT_embeddings = pT_embeddings#*self.pi
        
        pH_embeddings = (pH_embeddings+(pH_embeddings*pTime_embedding))*self.pi
        pT_embeddings = (pT_embeddings+(pT_embeddings*pTime_embedding))*self.pi

        nH_embeddings = nH_embeddings#*self.pi
        nR_embeddings = nR_embeddings*self.pi
        nT_embeddings = nT_embeddings#*self.pi
        
        nH_embeddings = (nH_embeddings+(nH_embeddings*nTime_embedding))*self.pi
        nT_embeddings = (nT_embeddings+(nT_embeddings*nTime_embedding))*self.pi

        positiveLoss_phase = torch.norm(torch.sin((pH_embeddings + pR_embeddings - pT_embeddings)/2), self.norm, 1)
        negativeLoss_phase = torch.norm(torch.sin((nH_embeddings + nR_embeddings- nT_embeddings)/2), self.norm, 1)
        #print(positiveLoss_phase)

        positiveLoss_mod = torch.norm(H_mod_embeddings*R_mod_embeddings-pStep_embedding,self.norm_m,1)
        negativeLoss_mod = torch.norm(H_mod_embeddings*R_mod_embeddings-nStep_embedding,self.norm_m,1)

        positiveLoss = self.hyper_p*positiveLoss_phase+self.hyper_m*positiveLoss_mod
        negativeLoss = self.hyper_p*negativeLoss_phase+self.hyper_m*negativeLoss_mod

        return positiveLoss, negativeLoss
    
    def Validate_entity_H(self,validateHead,validateRelation,validateTail,validateTime,validateStepH,trainTriple,numOfvalidateTriple):
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits1=0.0
        #print(numOfvalidateTriple)
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfvalidateTriple):
            validateHeadEmbedding = self.entity_embeddings(validateHead[i])
            validateRelationEmbedding = self.relation_embeddings(validateRelation[i])
            validateTailEmbedding = self.entity_embeddings(validateTail[i])
            validateTimeEmbedding = self.time_embeddings(validateTime[i])
            
            validateHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(validateHead[i]))
            validateRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(validateRelation[i]))
            validateTailEmbedding_mod = torch.abs(self.mod_e_embeddings(validateTail[i]))
            validatestep_embedding = torch.index_select(step_embeddings,0,validateStepH[i])

            validateHeadEmbedding = validateHeadEmbedding#*self.pi
            validateRelationEmbedding = validateRelationEmbedding*self.pi
            validateTailEmbedding = validateTailEmbedding#*self.pi
            
            validateHeadEmbedding = (validateHeadEmbedding+(validateHeadEmbedding*validateTimeEmbedding))*self.pi
            validateTailEmbedding = (validateTailEmbedding+(validateTailEmbedding*validateTimeEmbedding))*self.pi
            
            targetLoss_phase = torch.norm(torch.sin((validateHeadEmbedding + validateRelationEmbedding- validateTailEmbedding)/2), self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(validateHeadEmbedding_mod*validateRelationEmbedding_mod-validatestep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_p*targetLoss_phase+self.hyper_m*targetLoss_mod

            tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfEntity,1)
            
            tmpRelationEmbedding_mod = validateRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpTailEmbedding_mod = validateTailEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = validatestep_embedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = validateTimeEmbedding.repeat(self.numOfEntity,1)
            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
           
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))*self.pi
            
            tmpLoss_phase = torch.norm(torch.sin((entity_embeddings+tmpRelationEmbedding-tmpTailEmbedding)/2),self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding,self.norm_m,1).view(-1,1)
            
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            Rank_H=wrongHead.size()[0]+1
            if Rank_H<=10:
                Hits10_Raw+=1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
        mean_Rank_Raw = mean_Rank_Raw/numOfvalidateTriple
        Hits10_Raw = Hits10_Raw/numOfvalidateTriple
        return mean_Rank_Raw, Hits10_Raw

    def Validate_entity_T(self,validateHead,validateRelation,validateTail,validateTime,validateStepT,trainTriple,numOfvalidateTriple):
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits1=0.0
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfvalidateTriple):
            validateHeadEmbedding = self.entity_embeddings(validateHead[i])
            validateRelationEmbedding = self.relation_embeddings(validateRelation[i])
            validateTailEmbedding = self.entity_embeddings(validateTail[i])
            validateTimeEmbedding = self.time_embeddings(validateTime[i])
            
            validateTailEmbedding_mod = torch.abs(self.mod_e_embeddings(validateTail[i]))
            validateRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(validateRelation[i]))
            validateHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(validateHead[i]))
            validatestep_embedding = torch.index_select(step_embeddings,0,validateStepT[i])
            
            validateHeadEmbedding = validateHeadEmbedding#*self.pi
            validateRelationEmbedding = validateRelationEmbedding*self.pi
            validateTailEmbedding = validateTailEmbedding#*self.pi

            validateHeadEmbedding = (validateHeadEmbedding+(validateHeadEmbedding*validateTimeEmbedding))*self.pi
            validateTailEmbedding = (validateTailEmbedding+(validateTailEmbedding*validateTimeEmbedding))*self.pi

            targetLoss_phase = torch.norm(torch.sin((validateHeadEmbedding + validateRelationEmbedding- validateTailEmbedding)/2), self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(validateTailEmbedding_mod*validateRelationEmbedding_mod-validatestep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding_mod = validateRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpHeadEmbedding_mod = validateHeadEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = validatestep_embedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = validateTimeEmbedding.repeat(self.numOfEntity,1)

            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))*self.pi

            tmpLoss_phase = torch.norm(torch.sin((tmpHeadEmbedding+tmpRelationEmbedding-entity_embeddings)/2),self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding ,self.norm_m,1).view(-1,1)
        
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))

            Rank_H=wrongHead.size()[0]+1
            if Rank_H<=10:
                Hits10_Raw+=1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
        mean_Rank_Raw = mean_Rank_Raw/numOfvalidateTriple
        Hits10_Raw = Hits10_Raw/numOfvalidateTriple
        return mean_Rank_Raw, Hits10_Raw

    def fastValidate_relation(self,validateHead, validateRelation, validateTail,validateTime,validateStepH,numOfvalidateTriple):
        mean_Rank = 0.0
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfvalidateTriple):

            validateHeadEmbedding = self.entity_embeddings(validateHead[i])
            validateRelationEmbedding = self.relation_embeddings(validateRelation[i])
            validateTailEmbedding = self.entity_embeddings(validateTail[i])
            validateTimeEmbedding = self.time_embeddings(validateTime[i])
            
            validateHeadEmbedding_mod  = torch.abs(self.mod_e_embeddings(validateHead[i]))
            validateRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(validateRelation[i]))
            validateTailEmbedding_mod = torch.abs(self.mod_e_embeddings(validateTail[i]))
            
            validatestep_embedding = torch.index_select(step_embeddings,0,validateStepH[i])

            validateHeadEmbedding = validateHeadEmbedding#*self.pi
            validateRelationEmbedding  = validateRelationEmbedding*self.pi
            validateTailEmbedding = validateTailEmbedding#*self.pi
            
            validateHeadEmbedding = (validateHeadEmbedding+(validateHeadEmbedding*validateTimeEmbedding))*self.pi
            validateTailEmbedding = (validateTailEmbedding+(validateTailEmbedding*validateTimeEmbedding))*self.pi
            
            targetLoss_phase = torch.norm(torch.sin((validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding)/2), self.norm).repeat(self.numOfRelation, 1)
            targetLoss_mod = torch.norm(validateHeadEmbedding_mod*validateRelationEmbedding_mod-validatestep_embedding,self.norm_m).repeat(self.numOfRelation,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfRelation, 1)
            tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfRelation, 1)
            
            tmpHeadembedding_mod = validateHeadEmbedding_mod.repeat(self.numOfRelation,1)
            tmpTailEmbedding_mod = validateTailEmbedding_mod.repeat(self.numOfRelation,1)
            tmpstepEmbedding = validatestep_embedding.repeat(self.numOfRelation,1)

            tmpLoss_phase = torch.norm(torch.sin((tmpHeadEmbedding + self.relation_embeddings.weight.data*self.pi - tmpTailEmbedding)/2),
                                     self.norm, 1).view(-1, 1)
            tmpLoss_mod = torch.norm(tmpHeadembedding_mod*torch.abs(self.mod_r_embeddings.weight.data)-tmpstepEmbedding,self.norm_m,1).view(-1,1)

            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            rankR = wrongHead.size()[0]+1
            mean_Rank+=rankR
        return mean_Rank/numOfvalidateTriple
    
    def test_relation(self,testHead,testRelation,testTail,testTime,testStepH,trainTriple,numOfTestTriple):
        MRR=0.0
        MRR_F=0.0
        mean_Rank=0.0
        Hits1=0.0 
        mean_Rank_f=0.0
        Hits1_f=0.0
        
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            testTimeEmbedding = self.time_embeddings(testTime[i])
            
            testHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(testHead[i]))
            testRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(testRelation[i]))
            testTailEmbedding_mod = torch.abs(self.mod_e_embeddings(testTail[i]))
            teststep_embedding = torch.index_select(step_embeddings,0,testStepH[i])
            
            testHeadEmbedding = testHeadEmbedding#*self.pi
            testRelationEmbedding = testRelationEmbedding*self.pi
            testTailEmbedding = testTailEmbedding#*self.pi

            testHeadEmbedding = (testHeadEmbedding+(testHeadEmbedding*testTimeEmbedding))*self.pi
            testTailEmbedding = (testTailEmbedding+(testTailEmbedding*testTimeEmbedding))*self.pi
            
            targetLoss_phase = torch.norm(torch.sin((testHeadEmbedding + testRelationEmbedding- testTailEmbedding)/2), self.norm).repeat(self.numOfRelation, 1)
            targetLoss_mod = torch.norm(testHeadEmbedding_mod*testRelationEmbedding_mod-teststep_embedding,self.norm_m).repeat(self.numOfRelation,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            tmpHeadEmbedding = testHeadEmbedding.repeat(self.numOfRelation, 1)
            tmpTailEmbedding = testTailEmbedding.repeat(self.numOfRelation, 1)
            
            tmpHeadEmbedding_mod = testHeadEmbedding_mod.repeat(self.numOfRelation,1)
            tmpTailEmbedding_mod = testTailEmbedding_mod.repeat(self.numOfRelation,1)
            tmpstep_embedding_mod = teststep_embedding.repeat(self.numOfRelation,1)
            
            tmpLoss_phase=torch.norm(torch.sin((tmpHeadEmbedding+self.relation_embeddings.weight.data*self.pi-tmpTailEmbedding)/2),self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(tmpHeadEmbedding_mod*torch.abs(self.mod_r_embeddings.weight.data)-tmpstep_embedding_mod,self.norm_m,1).view(-1,1)
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongRelation = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            rankR = wrongRelation.size()[0]+1
            numOfFilterRelation=0
            for tmpWrongRelation in wrongRelation:
            #print(tmpWrongHead[0])
                numOfFilterRelation += trainTriple[(trainTriple[:,0]==testHead[i].float())&(trainTriple[:,1]==tmpWrongRelation[0].float())&(trainTriple[:,2]==testTail[i].float())&(trainTriple[:, 3]==testTime[i].float())].size()[0]
                
            rankR_f=max(rankR-numOfFilterRelation,1)
            mean_Rank_f=mean_Rank_f+rankR_f
            mean_Rank=mean_Rank+rankR
            MRR = MRR+float(1/rankR)
            MRR_F = MRR_F+float(1/rankR_f)
            if rankR==1:
                Hits1=Hits1+1
            if rankR_f==1:
                Hits1_f=Hits1_f+1
        mean_Rank_f = mean_Rank_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        mean_Rank = mean_Rank/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        MRR = MRR/numOfTestTriple
        MRR_F = MRR_F/numOfTestTriple
        
        return mean_Rank,Hits1,mean_Rank_f,Hits1_f, MRR, MRR_F
    
    def test_entity_H(self,testHead,testRelation,testTail,testTime,testStepH,trainTriple,numOfTestTriple):
        MRR=0.0
        MRR_F=0.0
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits3_f=0.0
        Hits1=0.0
        Hits1_f=0.0
        #print(numOfTestTriple)
        
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            testTimeEmbedding = self.time_embeddings(testTime[i])
            
            testHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(testHead[i]))
            testTailEmbedding_mod = torch.abs(self.mod_e_embeddings(testTail[i]))
            testRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(testRelation[i]))
            teststep_embedding = torch.index_select(step_embeddings,0,testStepH[i])
            
            testHeadEmbedding = testHeadEmbedding#*self.pi
            testRelationEmbedding = testRelationEmbedding*self.pi
            testTailEmbedding = testTailEmbedding#*self.pi
            testHeadEmbedding = (testHeadEmbedding+(testHeadEmbedding*testTimeEmbedding))*self.pi
            testTailEmbedding = (testTailEmbedding+(testTailEmbedding*testTimeEmbedding))*self.pi
           
            targetLoss_phase = torch.norm(torch.sin((testHeadEmbedding + testRelationEmbedding- testTailEmbedding)/2), self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(testHeadEmbedding_mod*testRelationEmbedding_mod-teststep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpRelationEmbedding = testRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTailEmbedding = testTailEmbedding.repeat(self.numOfEntity,1)
            
            tmpRelationEmbedding_mod = testRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpTailEmbedding_mod = testTailEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = teststep_embedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = testTimeEmbedding.repeat(self.numOfEntity,1)

            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))*self.pi
            
            tmpLoss_phase = torch.norm(torch.sin((entity_embeddings+tmpRelationEmbedding-tmpTailEmbedding)/2),self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding,self.norm_m,1).view(-1,1)
            
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            Rank_H=wrongHead.size()[0]+1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
            MRR = MRR +float(1/Rank_H)
            if Rank_H<=10:
                Hits10_Raw=Hits10_Raw+1
            if Rank_H<=3:
                Hits3=Hits3+1
            if Rank_H==1:
                Hits1=Hits1+1
                
            numOfFilterHead=0
            for tmpWrongHead in wrongHead:
                #print(tmpWrongHead)
                numOfFilterHead += trainTriple[(trainTriple[:,0]==tmpWrongHead[0].float())&(trainTriple[:,1]==testRelation[i].float())&(trainTriple[:,2]==testTail[i].float())&(trainTriple[:, 3]==testTime[i].float())].size()[0]
            
            Rank_H_filter=max(Rank_H-numOfFilterHead,1)
            
            mean_Rank_filter=mean_Rank_filter+Rank_H_filter
            MRR_F = MRR_F+float(1/Rank_H_filter)
            if Rank_H_filter<=10:
                Hits10_filter=Hits10_filter+1
        #print(len(wrongTail))
        if Rank_H_filter<=3:
            Hits3_f=Hits3_f+1
        if Rank_H_filter<=1:
            Hits1_f=Hits1_f+1
        #print(len(wrongTail))
            
        Hits10_Raw=Hits10_Raw/numOfTestTriple
        Hits10_filter=Hits10_filter/numOfTestTriple
        mean_Rank_Raw=mean_Rank_Raw/numOfTestTriple
        mean_Rank_filter=mean_Rank_filter/numOfTestTriple
        Hits3 = Hits3/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        Hits3_f = Hits3_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        MRR = MRR/numOfTestTriple
        MRR_F = MRR_F/numOfTestTriple
        
        return mean_Rank_Raw, Hits10_Raw, mean_Rank_filter, Hits10_filter,Hits3, Hits1, Hits3_f, Hits1_f,MRR, MRR_F
    
    def test_entity_T(self,testHead,testRelation,testTail,testTime,testStepT,trainTriple,numOfTestTriple):
        MRR=0.0
        MRR_F=0.0
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits3_f=0.0
        Hits1=0.0
        Hits1_f=0.0
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            testTimeEmbedding = self.time_embeddings(testTime[i])
            
            testTailEmbedding_mod = torch.abs(self.mod_e_embeddings(testTail[i]))
            testRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(testRelation[i]))
            testHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(testHead[i]))
            teststep_embedding = torch.index_select(step_embeddings,0,testStepT[i])
            
            testHeadEmbedding = testHeadEmbedding#*self.pi
            testRelationEmbedding = testRelationEmbedding*self.pi
            testTailEmbedding = testTailEmbedding#*self.pi
            testHeadEmbedding = (testHeadEmbedding+(testHeadEmbedding*testTimeEmbedding))*self.pi
            testTailEmbedding = (testTailEmbedding+(testTailEmbedding*testTimeEmbedding))*self.pi
            
            targetLoss_phase = torch.norm(torch.sin((testHeadEmbedding + testRelationEmbedding- testTailEmbedding)/2), self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(testTailEmbedding_mod*testRelationEmbedding_mod-teststep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpHeadEmbedding = testHeadEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding = testRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = testTimeEmbedding.repeat(self.numOfEntity,1)

            tmpRelationEmbedding_mod = testRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpHeadembedding_mod = testHeadEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = teststep_embedding.repeat(self.numOfEntity,1)

            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))*self.pi
            
            tmpLoss_phase = torch.norm(torch.sin((tmpHeadEmbedding+tmpRelationEmbedding-entity_embeddings)/2),self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding,self.norm_m,1).view(-1,1)
            
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            Rank_H=wrongHead.size()[0]+1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
            MRR = MRR+float(1/Rank_H)
            if Rank_H<=10:
                Hits10_Raw=Hits10_Raw+1
            if Rank_H<=3:
                Hits3=Hits3+1
            if Rank_H==1:
                Hits1=Hits1+1
                
            numOfFilterHead=0
            for tmpWrongHead in wrongHead:
                #print(tmpWrongHead)
                numOfFilterHead += trainTriple[(trainTriple[:,0]==testHead[i].float())&(trainTriple[:,1]==testRelation[i].float())&(trainTriple[:,2]==tmpWrongHead[0].float())&(trainTriple[:, 3]==testTime[i].float())].size()[0]
            
            Rank_H_filter=max(Rank_H-numOfFilterHead,1)
            
            mean_Rank_filter=mean_Rank_filter+Rank_H_filter
            MRR_F=MRR_F+float(1/Rank_H_filter)
            if Rank_H_filter<=10:
                Hits10_filter=Hits10_filter+1
            if Rank_H_filter<=3:
                Hits3_f=Hits3_f+1
            if Rank_H_filter<=1:
                Hits1_f=Hits1_f+1
        #print(len(wrongTail))
            
        Hits10_Raw=Hits10_Raw/numOfTestTriple
        Hits10_filter=Hits10_filter/numOfTestTriple
        mean_Rank_Raw=mean_Rank_Raw/numOfTestTriple
        mean_Rank_filter=mean_Rank_filter/numOfTestTriple
        Hits3 = Hits3/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        Hits3_f = Hits3_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        MRR = MRR/numOfTestTriple
        MRR_F = MRR_F/numOfTestTriple
        
        return mean_Rank_Raw, Hits10_Raw, mean_Rank_filter, Hits10_filter,Hits3, Hits1, Hits3_f, Hits1_f,MRR, MRR_F

class MPKE_Trans(nn.Module):
    
    def __init__(self, numOfEntity, numOfRelation, numOfTime, numOfMaxLen, entityDimension, relationDimension, norm):
        super(MPKE_Trans,self).__init__()

        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.numOfTime = numOfTime
        self.numOfMaxLen = numOfMaxLen
        self.entityDimension = entityDimension
        self.relationDimension = relationDimension
        self.norm = norm
        
        self.pi = 3.14159262358979323846

        self.relation_embeddings = nn.Embedding(self.numOfRelation, self.relationDimension)
        nn.init.xavier_normal_(
            tensor=self.relation_embeddings.weight.data
        )
        
        self.mod_r_embeddings = nn.Embedding(self.numOfRelation, self.relationDimension)
        nn.init.xavier_normal_(
            tensor=self.relation_embeddings.weight.data
        )
        self.entity_embeddings = nn.Embedding(self.numOfEntity, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.entity_embeddings.weight.data
        )
        
        self.mod_e_embeddings = nn.Embedding(self.numOfEntity, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.entity_embeddings.weight.data
        )
        
        self.time_embeddings = nn.Embedding(self.numOfTime, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.entity_embeddings.weight.data
        )
        
        self.step_embeddings = nn.Embedding(self.numOfMaxLen, self.entityDimension)
        nn.init.xavier_normal_(
            tensor=self.step_embeddings.weight.data
        )
        '''
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data,1,1)
        self.mod_r_embeddings.weight.data = F.normalize(self.mod_r_embeddings.weight.data,1,1)
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data,1,1)
        self.mod_e_embeddings.weight.data = F.normalize(self.mod_e_embeddings.weight.data,1,1)
        self.time_embeddings.weight.data = F.normalize(self.time_embeddings.weight.data,1,1)
        self.step_embeddings.weight.data = F.normalize(self.step_embeddings.weight.data,1,1)
        '''
        self.hyper_p=1
        self.hyper_m=0.5#0.01
        self.norm_m=2
        

    def forward(self, positiveBatchHead, positiveBatchRelation, positiveBatchTail,positiveBatchTime, positiveBatchStep,corruptedBatchHead, corruptedBatchRelation, corruptedBatchTail, corruptedBatchTime, corruptedBatchStep):

        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        #print(corruptedBatchTime)
        pH_embeddings = self.entity_embeddings(positiveBatchHead)
        pR_embeddings = self.relation_embeddings(positiveBatchRelation)
        pT_embeddings = self.entity_embeddings(positiveBatchTail)
        pTime_embedding = self.time_embeddings(positiveBatchTime)
        #print(corruptedBatchStep)
        pStep_embedding = torch.index_select(step_embeddings,0,positiveBatchStep)

        nH_embeddings = self.entity_embeddings(corruptedBatchHead)
        nR_embeddings = self.relation_embeddings(corruptedBatchRelation)
        nT_embeddings = self.entity_embeddings(corruptedBatchTail)
        nTime_embedding = self.time_embeddings(corruptedBatchTime)
        #print(len(step_embeddings))
        nStep_embedding = torch.index_select(step_embeddings,0,corruptedBatchStep)

        H_mod_embeddings = torch.abs(self.mod_e_embeddings(positiveBatchHead))
        R_mod_embeddings = torch.abs(self.mod_r_embeddings(positiveBatchRelation))
        T_mod_embeddings = torch.abs(self.mod_e_embeddings(positiveBatchTail))

        
        pH_embeddings = (pH_embeddings+(pH_embeddings*pTime_embedding))
        pT_embeddings = (pT_embeddings+(pT_embeddings*pTime_embedding))

        
        nH_embeddings = (nH_embeddings+(nH_embeddings*nTime_embedding))
        nT_embeddings = (nT_embeddings+(nT_embeddings*nTime_embedding))

        positiveLoss_phase = torch.norm(pH_embeddings + pR_embeddings - pT_embeddings, self.norm, 1)
        negativeLoss_phase = torch.norm(nH_embeddings + nR_embeddings- nT_embeddings, self.norm, 1)
        #print(positiveLoss_phase)

        positiveLoss_mod = torch.norm(H_mod_embeddings*R_mod_embeddings-pStep_embedding,self.norm_m,1)
        negativeLoss_mod = torch.norm(H_mod_embeddings*R_mod_embeddings-nStep_embedding,self.norm_m,1)

        positiveLoss = self.hyper_p*positiveLoss_phase+self.hyper_m*positiveLoss_mod
        negativeLoss = self.hyper_p*negativeLoss_phase+self.hyper_m*negativeLoss_mod

        return positiveLoss, negativeLoss
    
    def Validate_entity_H(self,validateHead,validateRelation,validateTail,validateTime,validateStepH,trainTriple,numOfvalidateTriple):
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits1=0.0
        #print(numOfvalidateTriple)
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfvalidateTriple):
            validateHeadEmbedding = self.entity_embeddings(validateHead[i])
            validateRelationEmbedding = self.relation_embeddings(validateRelation[i])
            validateTailEmbedding = self.entity_embeddings(validateTail[i])
            validateTimeEmbedding = self.time_embeddings(validateTime[i])
            
            validateHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(validateHead[i]))
            validateRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(validateRelation[i]))
            validateTailEmbedding_mod = torch.abs(self.mod_e_embeddings(validateTail[i]))
            validatestep_embedding = torch.index_select(step_embeddings,0,validateStepH[i])
            
            validateHeadEmbedding = (validateHeadEmbedding+(validateHeadEmbedding*validateTimeEmbedding))
            validateTailEmbedding = (validateTailEmbedding+(validateTailEmbedding*validateTimeEmbedding))
            
            targetLoss_phase = torch.norm(validateHeadEmbedding + validateRelationEmbedding- validateTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(validateHeadEmbedding_mod*validateRelationEmbedding_mod-validatestep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_p*targetLoss_phase+self.hyper_m*targetLoss_mod

            tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfEntity,1)
            
            tmpRelationEmbedding_mod = validateRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpTailEmbedding_mod = validateTailEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = validatestep_embedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = validateTimeEmbedding.repeat(self.numOfEntity,1)
            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
           
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))
            
            tmpLoss_phase = torch.norm(entity_embeddings+tmpRelationEmbedding-tmpTailEmbedding,self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding,self.norm_m,1).view(-1,1)
            
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            Rank_H=wrongHead.size()[0]+1
            if Rank_H<=10:
                Hits10_Raw+=1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
        mean_Rank_Raw = mean_Rank_Raw/numOfvalidateTriple
        Hits10_Raw = Hits10_Raw/numOfvalidateTriple
        return mean_Rank_Raw, Hits10_Raw

    def Validate_entity_T(self,validateHead,validateRelation,validateTail,validateTime,validateStepT,trainTriple,numOfvalidateTriple):
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits1=0.0
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfvalidateTriple):
            validateHeadEmbedding = self.entity_embeddings(validateHead[i])
            validateRelationEmbedding = self.relation_embeddings(validateRelation[i])
            validateTailEmbedding = self.entity_embeddings(validateTail[i])
            validateTimeEmbedding = self.time_embeddings(validateTime[i])
            
            validateTailEmbedding_mod = torch.abs(self.mod_e_embeddings(validateTail[i]))
            validateRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(validateRelation[i]))
            validateHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(validateHead[i]))
            validatestep_embedding = torch.index_select(step_embeddings,0,validateStepT[i])

            validateHeadEmbedding = (validateHeadEmbedding+(validateHeadEmbedding*validateTimeEmbedding))
            validateTailEmbedding = (validateTailEmbedding+(validateTailEmbedding*validateTimeEmbedding))

            targetLoss_phase = torch.norm(validateHeadEmbedding + validateRelationEmbedding- validateTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(validateTailEmbedding_mod*validateRelationEmbedding_mod-validatestep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding_mod = validateRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpHeadEmbedding_mod = validateHeadEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = validatestep_embedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = validateTimeEmbedding.repeat(self.numOfEntity,1)

            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))

            tmpLoss_phase = torch.norm(tmpHeadEmbedding+tmpRelationEmbedding-entity_embeddings,self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding ,self.norm_m,1).view(-1,1)
        
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))

            Rank_H=wrongHead.size()[0]+1
            if Rank_H<=10:
                Hits10_Raw+=1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
        mean_Rank_Raw = mean_Rank_Raw/numOfvalidateTriple
        Hits10_Raw = Hits10_Raw/numOfvalidateTriple
        return mean_Rank_Raw, Hits10_Raw

    def fastValidate_relation(self,validateHead, validateRelation, validateTail,validateTime,validateStepH,numOfvalidateTriple):
        mean_Rank = 0.0
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfvalidateTriple):

            validateHeadEmbedding = self.entity_embeddings(validateHead[i])
            validateRelationEmbedding = self.relation_embeddings(validateRelation[i])
            validateTailEmbedding = self.entity_embeddings(validateTail[i])
            validateTimeEmbedding = self.time_embeddings(validateTime[i])
            
            validateHeadEmbedding_mod  = torch.abs(self.mod_e_embeddings(validateHead[i]))
            validateRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(validateRelation[i]))
            validateTailEmbedding_mod = torch.abs(self.mod_e_embeddings(validateTail[i]))
            
            validatestep_embedding = torch.index_select(step_embeddings,0,validateStepH[i])

            
            validateHeadEmbedding = (validateHeadEmbedding+(validateHeadEmbedding*validateTimeEmbedding))
            validateTailEmbedding = (validateTailEmbedding+(validateTailEmbedding*validateTimeEmbedding))
            
            targetLoss_phase = torch.norm(validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding, self.norm).repeat(self.numOfRelation, 1)
            targetLoss_mod = torch.norm(validateHeadEmbedding_mod*validateRelationEmbedding_mod-validatestep_embedding,self.norm_m).repeat(self.numOfRelation,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfRelation, 1)
            tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfRelation, 1)
            
            tmpHeadembedding_mod = validateHeadEmbedding_mod.repeat(self.numOfRelation,1)
            tmpTailEmbedding_mod = validateTailEmbedding_mod.repeat(self.numOfRelation,1)
            tmpstepEmbedding = validatestep_embedding.repeat(self.numOfRelation,1)

            tmpLoss_phase = torch.norm(tmpHeadEmbedding + self.relation_embeddings.weight.data - tmpTailEmbedding,
                                     self.norm, 1).view(-1, 1)
            tmpLoss_mod = torch.norm(tmpHeadembedding_mod*torch.abs(self.mod_r_embeddings.weight.data)-tmpstepEmbedding,self.norm_m,1).view(-1,1)

            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            rankR = wrongHead.size()[0]+1
            mean_Rank+=rankR
        return mean_Rank/numOfvalidateTriple
    
    def test_relation(self,testHead,testRelation,testTail,testTime,testStepH,trainTriple,numOfTestTriple):
        MRR=0.0
        MRR_F=0.0
        mean_Rank=0.0
        Hits1=0.0 
        mean_Rank_f=0.0
        Hits1_f=0.0
        
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            testTimeEmbedding = self.time_embeddings(testTime[i])
            
            testHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(testHead[i]))
            testRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(testRelation[i]))
            testTailEmbedding_mod = torch.abs(self.mod_e_embeddings(testTail[i]))
            teststep_embedding = torch.index_select(step_embeddings,0,testStepH[i])
            

            testHeadEmbedding = (testHeadEmbedding+(testHeadEmbedding*testTimeEmbedding))
            testTailEmbedding = (testTailEmbedding+(testTailEmbedding*testTimeEmbedding))
            
            targetLoss_phase = torch.norm(testHeadEmbedding + testRelationEmbedding- testTailEmbedding, self.norm).repeat(self.numOfRelation, 1)
            targetLoss_mod = torch.norm(testHeadEmbedding_mod*testRelationEmbedding_mod-teststep_embedding,self.norm_m).repeat(self.numOfRelation,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            tmpHeadEmbedding = testHeadEmbedding.repeat(self.numOfRelation, 1)
            tmpTailEmbedding = testTailEmbedding.repeat(self.numOfRelation, 1)
            
            tmpHeadEmbedding_mod = testHeadEmbedding_mod.repeat(self.numOfRelation,1)
            tmpTailEmbedding_mod = testTailEmbedding_mod.repeat(self.numOfRelation,1)
            tmpstep_embedding_mod = teststep_embedding.repeat(self.numOfRelation,1)
            
            tmpLoss_phase=torch.norm(tmpHeadEmbedding+self.relation_embeddings.weight.data-tmpTailEmbedding,self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(tmpHeadEmbedding_mod*torch.abs(self.mod_r_embeddings.weight.data)-tmpstep_embedding_mod,self.norm_m,1).view(-1,1)
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongRelation = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            rankR = wrongRelation.size()[0]+1
            numOfFilterRelation=0
            for tmpWrongRelation in wrongRelation:
            #print(tmpWrongHead[0])
                numOfFilterRelation += trainTriple[(trainTriple[:,0]==testHead[i].float())&(trainTriple[:,1]==tmpWrongRelation[0].float())&(trainTriple[:,2]==testTail[i].float())&(trainTriple[:, 3]==testTime[i].float())].size()[0]
                
            rankR_f=max(rankR-numOfFilterRelation,1)
            mean_Rank_f=mean_Rank_f+rankR_f
            mean_Rank=mean_Rank+rankR
            MRR = MRR+float(1/rankR)
            MRR_F = MRR_F+float(1/rankR_f)
            if rankR==1:
                Hits1=Hits1+1
            if rankR_f==1:
                Hits1_f=Hits1_f+1
        mean_Rank_f = mean_Rank_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        mean_Rank = mean_Rank/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        MRR = MRR/numOfTestTriple
        MRR_F = MRR_F/numOfTestTriple
        
        return mean_Rank,Hits1,mean_Rank_f,Hits1_f, MRR, MRR_F
    
    def test_entity_H(self,testHead,testRelation,testTail,testTime,testStepH,trainTriple,numOfTestTriple):
        MRR=0.0
        MRR_F=0.0
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits3_f=0.0
        Hits1=0.0
        Hits1_f=0.0
        #print(numOfTestTriple)
        
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            testTimeEmbedding = self.time_embeddings(testTime[i])
            
            testHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(testHead[i]))
            testTailEmbedding_mod = torch.abs(self.mod_e_embeddings(testTail[i]))
            testRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(testRelation[i]))
            teststep_embedding = torch.index_select(step_embeddings,0,testStepH[i])
            
            testHeadEmbedding = (testHeadEmbedding+(testHeadEmbedding*testTimeEmbedding))
            testTailEmbedding = (testTailEmbedding+(testTailEmbedding*testTimeEmbedding))
           
            targetLoss_phase = torch.norm(testHeadEmbedding + testRelationEmbedding- testTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(testHeadEmbedding_mod*testRelationEmbedding_mod-teststep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpRelationEmbedding = testRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTailEmbedding = testTailEmbedding.repeat(self.numOfEntity,1)
            
            tmpRelationEmbedding_mod = testRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpTailEmbedding_mod = testTailEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = teststep_embedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = testTimeEmbedding.repeat(self.numOfEntity,1)

            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))
            
            tmpLoss_phase = torch.norm(entity_embeddings+tmpRelationEmbedding-tmpTailEmbedding,self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding,self.norm_m,1).view(-1,1)
            
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            Rank_H=wrongHead.size()[0]+1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
            MRR = MRR +float(1/Rank_H)
            if Rank_H<=10:
                Hits10_Raw=Hits10_Raw+1
            if Rank_H<=3:
                Hits3=Hits3+1
            if Rank_H==1:
                Hits1=Hits1+1
                
            numOfFilterHead=0
            for tmpWrongHead in wrongHead:
                #print(tmpWrongHead)
                numOfFilterHead += trainTriple[(trainTriple[:,0]==tmpWrongHead[0].float())&(trainTriple[:,1]==testRelation[i].float())&(trainTriple[:,2]==testTail[i].float())&(trainTriple[:, 3]==testTime[i].float())].size()[0]
            
            Rank_H_filter=max(Rank_H-numOfFilterHead,1)
            
            mean_Rank_filter=mean_Rank_filter+Rank_H_filter
            MRR_F = MRR_F+float(1/Rank_H_filter)
            if Rank_H_filter<=10:
                Hits10_filter=Hits10_filter+1
        #print(len(wrongTail))
        if Rank_H_filter<=3:
            Hits3_f=Hits3_f+1
        if Rank_H_filter<=1:
            Hits1_f=Hits1_f+1
        #print(len(wrongTail))
            
        Hits10_Raw=Hits10_Raw/numOfTestTriple
        Hits10_filter=Hits10_filter/numOfTestTriple
        mean_Rank_Raw=mean_Rank_Raw/numOfTestTriple
        mean_Rank_filter=mean_Rank_filter/numOfTestTriple
        Hits3 = Hits3/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        Hits3_f = Hits3_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        MRR = MRR/numOfTestTriple
        MRR_F = MRR_F/numOfTestTriple
        
        return mean_Rank_Raw, Hits10_Raw, mean_Rank_filter, Hits10_filter,Hits3, Hits1, Hits3_f, Hits1_f,MRR, MRR_F
    
    def test_entity_T(self,testHead,testRelation,testTail,testTime,testStepT,trainTriple,numOfTestTriple):
        MRR=0.0
        MRR_F=0.0
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        Hits3=0.0
        Hits3_f=0.0
        Hits1=0.0
        Hits1_f=0.0
        step_embeddings = torch.cumsum(torch.abs(self.step_embeddings.weight.data),0)
        
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            testTimeEmbedding = self.time_embeddings(testTime[i])
            
            testTailEmbedding_mod = torch.abs(self.mod_e_embeddings(testTail[i]))
            testRelationEmbedding_mod = torch.abs(self.mod_r_embeddings(testRelation[i]))
            testHeadEmbedding_mod = torch.abs(self.mod_e_embeddings(testHead[i]))
            teststep_embedding = torch.index_select(step_embeddings,0,testStepT[i])

            testHeadEmbedding = (testHeadEmbedding+(testHeadEmbedding*testTimeEmbedding))
            testTailEmbedding = (testTailEmbedding+(testTailEmbedding*testTimeEmbedding))
            
            targetLoss_phase = torch.norm(testHeadEmbedding + testRelationEmbedding- testTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
            targetLoss_mod = torch.norm(testTailEmbedding_mod*testRelationEmbedding_mod-teststep_embedding,self.norm_m).repeat(self.numOfEntity,1)
            
            targetLoss = self.hyper_m*targetLoss_mod+self.hyper_p*targetLoss_phase
            
            tmpHeadEmbedding = testHeadEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding = testRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTimeEmbedding = testTimeEmbedding.repeat(self.numOfEntity,1)

            tmpRelationEmbedding_mod = testRelationEmbedding_mod.repeat(self.numOfEntity,1)
            tmpHeadembedding_mod = testHeadEmbedding_mod.repeat(self.numOfEntity,1)
            tmpstepEmbedding = teststep_embedding.repeat(self.numOfEntity,1)

            entity_embeddings = self.entity_embeddings.weight.data#*self.pi
            entity_embeddings = (entity_embeddings+(entity_embeddings*tmpTimeEmbedding))
            
            tmpLoss_phase = torch.norm(tmpHeadEmbedding+tmpRelationEmbedding-entity_embeddings,self.norm,1).view(-1,1)
            tmpLoss_mod = torch.norm(torch.abs(self.mod_e_embeddings.weight.data)*tmpRelationEmbedding_mod-tmpstepEmbedding,self.norm_m,1).view(-1,1)
            
            tmpLoss = self.hyper_m*tmpLoss_mod+self.hyper_p*tmpLoss_phase

            wrongHead = torch.nonzero(nn.functional.relu(targetLoss - tmpLoss))
            Rank_H=wrongHead.size()[0]+1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H
            MRR = MRR+float(1/Rank_H)
            if Rank_H<=10:
                Hits10_Raw=Hits10_Raw+1
            if Rank_H<=3:
                Hits3=Hits3+1
            if Rank_H==1:
                Hits1=Hits1+1
                
            numOfFilterHead=0
            for tmpWrongHead in wrongHead:
                #print(tmpWrongHead)
                numOfFilterHead += trainTriple[(trainTriple[:,0]==testHead[i].float())&(trainTriple[:,1]==testRelation[i].float())&(trainTriple[:,2]==tmpWrongHead[0].float())&(trainTriple[:, 3]==testTime[i].float())].size()[0]
            
            Rank_H_filter=max(Rank_H-numOfFilterHead,1)
            
            mean_Rank_filter=mean_Rank_filter+Rank_H_filter
            MRR_F=MRR_F+float(1/Rank_H_filter)
            if Rank_H_filter<=10:
                Hits10_filter=Hits10_filter+1
            if Rank_H_filter<=3:
                Hits3_f=Hits3_f+1
            if Rank_H_filter<=1:
                Hits1_f=Hits1_f+1
        #print(len(wrongTail))
            
        Hits10_Raw=Hits10_Raw/numOfTestTriple
        Hits10_filter=Hits10_filter/numOfTestTriple
        mean_Rank_Raw=mean_Rank_Raw/numOfTestTriple
        mean_Rank_filter=mean_Rank_filter/numOfTestTriple
        Hits3 = Hits3/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        Hits3_f = Hits3_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        MRR = MRR/numOfTestTriple
        MRR_F = MRR_F/numOfTestTriple
        
        return mean_Rank_Raw, Hits10_Raw, mean_Rank_filter, Hits10_filter,Hits3, Hits1, Hits3_f, Hits1_f,MRR, MRR_F