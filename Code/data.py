import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import data
import h5py as hp
from Code.tangent_space import make_symmetric
import pandas as pd

#Custom dataset for UKB FNN
class UKB_FMRI_Dataset_FNN(data.Dataset):
    
    def __init__(self,data_dir,df):
        
        self.df = df
        self.data_dir = data_dir
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self,index):
       
        sub_id = self.df.loc[index ,'Subject']        
        labels = self.df.Gender[self.df.loc[self.df['Subject']==sub_id].index[0]]       
        X = np.loadtxt(self.data_dir+str(sub_id)+"/func/corr_matrix/"+str(sub_id)+"_rfMRI-partial-correlation-matrix-dimension-100.txt") 
        images = torch.FloatTensor(X) 
        labels = torch.tensor(labels, dtype=torch.float32)        
        labels = labels.unsqueeze_(-1)
        return images,labels
    
#Custom dataset for UKB BCNN
class UKB_FMRI_Dataset_BCNN(data.Dataset):
    
    def __init__(self,data_dir,df):
        
        self.df = df
        self.data_dir = data_dir
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self,index):
       
        sub_id = self.df.loc[index ,'Subject']        
        labels = self.df.Gender[self.df.loc[self.df['Subject']==sub_id].index[0]]       
        X = np.loadtxt(self.data_dir+str(sub_id)+"/func/corr_matrix/"+str(sub_id)+"_rfMRI-partial-correlation-matrix-dimension-100.txt") 
        X = make_symmetric(X)
        X = X[np.newaxis,:]
        images = torch.FloatTensor(X) 
        labels = torch.tensor(labels, dtype=torch.float32)        
        labels = labels.unsqueeze_(-1)
        return images,labels
    

def data_loader_UKB(data_dir,df):      
      
    data = []
    labels = []
    for sub in df.Subject:
        y = df.Gender[df.loc[df['Subject']==sub].index[0]]         
        X = np.loadtxt(data_dir+str(sub)+"/func/corr_matrix/"+str(sub)+"_rfMRI-partial-correlation-matrix-dimension-100.txt") 
        data.append(X)
        labels.append(y)
    return np.asarray(data),np.asarray(labels)



class HCP_Dataset_FNN(data.Dataset):
    
    def __init__(self,sub_list,path_sub,path_gender):
        
        self.sub_list = sub_list
        self.path_sub = path_sub
        self.path_gen = path_gender
        self.df = pd.read_excel(self.path_gen)
        
    def __len__(self):
        
        return len(self.sub_list)
    
    def __getitem__(self,index):
       
        sub_id = int(self.sub_list[index])
        y_to_num = {'M':1,'F': 0}
        labels = y_to_num[self.df.Gender[self.df.loc[self.df['Subject']==sub_id].index[0]]]         
        X  = hp.File(self.path_sub + str(sub_id)+'.mat' ,'r')['CorrMatrix'][()]               
        r,c = X.shape        
        inds = np.tril_indices(n=r,m=c) 
        vals = X[inds]
        images = torch.FloatTensor(vals) 
        labels = torch.tensor(labels, dtype=torch.float32)        
        labels = labels.unsqueeze_(-1)
        return images,labels


def data_loader_HCP(sub_list,path_gen,path_sub):
    
    df = pd.read_excel(path_gen)
    y_to_num = {'M':1,'F': 0}
    
    data = []
    labels = []
    for sub_id in sub_list:
        y = y_to_num[df.Gender[df.loc[df['Subject']==sub_id].index[0]]]         
        X = hp.File(path_sub + str(sub_id)+'.mat' ,'r')['CorrMatrix'][()]   
        r,c = X.shape        
        inds = np.tril_indices(n=r,m=c) 
        vals = X[inds]
        data.append(vals)
        labels.append(y)
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data,labels
 
class HCP_Dataset_BCNN(data.Dataset):
    
    def __init__(self,sub_list,path_sub,path_gender):
        
        self.sub_list = sub_list
        self.path_sub = path_sub
        self.path_gen = path_gender
        self.df = pd.read_excel(self.path_gen)
        
    def __len__(self):
        
        return len(self.sub_list)
    
    def __getitem__(self,index):
       
        sub_id = int(self.sub_list[index])
        y_to_num = {'M':1,'F': 0}
        labels = y_to_num[self.df.Gender[self.df.loc[self.df['Subject']==sub_id].index[0]]]         
        X  = hp.File(self.path_sub + str(sub_id)+'.mat' ,'r')['CorrMatrix'][()]               
        X = X[np.newaxis,:]
        images = torch.FloatTensor(X) 
        labels = torch.tensor(labels, dtype=torch.float32)        
        labels = labels.unsqueeze_(-1)
        return images,labels

