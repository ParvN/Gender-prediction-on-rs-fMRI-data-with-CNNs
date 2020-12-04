import os
import numpy as np
import pandas as pd
import sys
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, classification_report,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import data

nitorch_dir = "/nitorch/"
sys.path.insert(0, os.path.join(nitorch_dir))
from nitorch.metrics import binary_balanced_accuracy, sensitivity, specificity

#SVM model
def SVM(X_train,y_train,X_test,y_test,**kwargs) :
    clf = SVC(**kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def Elastic_Net(X_train,y_train,X_test,y_test,**kwargs) :
    clf = ElasticNet(**kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)    
    return y_pred

#Feed forward neural network for UKB
class FNN(nn.Module):
    
    def __init__(self,input_size,n_l1,n_l2,output_size,drop_out):
        super(FNN,self).__init__()
        
        self.fc1 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(input_size, n_l1),
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(n_l1),
                                 
                                 )
        
        self.fc2 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(n_l1,n_l2),
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(n_l2),
                                 
                                )
        
        self.fc3 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(n_l2,1),
                                                                  
                                )
        
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self,x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)       
        return x
    
#Feedforward neural network for HCP with an additional layer
class HCP_FNN(nn.Module):
    
    def __init__(self,input_size,n_l1,n_l2,n_l3,output_size,drop_out):
        super(HCP_FNN,self).__init__()
        
        self.fc1 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(input_size, n_l1),
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(n_l1),
                                 
                                 )
        
        self.fc2 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(n_l1,n_l2),
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(n_l2),
                                 
                                )
        self.fc3 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(n_l2,n_l3),
                                 
                                 nn.Sigmoid(),
                                 nn.BatchNorm1d(n_l3),
                                 
                                )
        
        self.fc4 = nn.Sequential(
                                 nn.Dropout(drop_out),
                                 nn.Linear(n_l3,1),
                                                                  
                                )
        
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self,x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)       
        x = self.fc4(x)  
        return x
    


# Implementation of BCNN from Tong He et al, https://www.biorxiv.org/content/10.1101/473603v1.full
# BrainNetCNN edge to edge layer
class Edge2Edge(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Edge, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))
 
    # implemented by two conv2d with line filter
    def forward(self, x):
        size = x.size()
        row = self.row_conv(x)
        col = self.col_conv(x)
        row_ex = row.expand(size[0], self.filters, self.dim, self.dim)
        col_ex = col.expand(size[0], self.filters, self.dim, self.dim)
        return row_ex + col_ex
        
 
 
# BrainNetCNN edge to node layer
class Edge2Node(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Node, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))
 
    def forward(self, x):
        row = self.row_conv(x)
        col = self.col_conv(x)
        return row + col.permute(0, 1, 3, 2)
 
 
# BrainNetCNN node to graph layer
class Node2Graph(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Node2Graph, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.conv = nn.Conv2d(channel, filters, (dim, 1))
 
    def forward(self, x):
        return self.conv(x)
 
 
# BrainNetCNN network
class BCNN(nn.Module):
    def __init__(self, e2e, e2n, n2g, f_size,dropout):
        super(BCNN, self).__init__()
        self.n2g_filter = n2g
        self.e2e = Edge2Edge(1, f_size, e2e)
        self.e2n = Edge2Node(e2e, f_size, e2n)
        self.dropout = nn.Dropout(p=dropout)
        self.n2g = Node2Graph(e2n, f_size, n2g)
        self.fc = nn.Linear(n2g, 1)
        self.BatchNorm = nn.BatchNorm1d(n2g)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
 
    def forward(self, x):
        x = self.e2e(x)
        x = self.dropout(x)
        x = self.e2n(x)
        x = self.dropout(x)
        x = self.n2g(x)
        x = self.dropout(x)
        x = x.view(-1, self.n2g_filter)
        x = self.fc(self.BatchNorm(x))
        
        return x

