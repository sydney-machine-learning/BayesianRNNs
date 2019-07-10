#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:55:14 2019

@author: ashrey
"""

import numpy as np
data = np.loadtxt('scaled_dataset_mackey.txt')
x=[]
y=[]
for i in range(0,len(data[0:-26])):
    t = np.random.randint(1,25)
    x.append(data[i:i+t])
    y.append(data[i+t])

f=open("mackey.txt","w+")
for i,sample in enumerate(x):
    f.write(str(len(sample))+"\n")
    for item in sample:
        f.write(str(item)+"\n")
    f.write(str(y[i])+"\n\n")

f.close()