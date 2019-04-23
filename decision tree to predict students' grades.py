# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:36:28 2019

@author: Neo Soon Hua
"""

pip install sklearn
pip install scipy  

from sklearn import tree
clf = tree.DecisionTreeClassifier()    
import pandas as pd

trainFilename="Decision Tree/train"
testFilename="Decision Tree/test"
xTrain = pd.read_csv(trainFilename+'.csv', usecols=['SH1 CT', 'T3LT', 'Promo', 'SH2 T1LT (%)', 'SH2 CT', 'Prelim'])
xTrain
yTrain = pd.read_csv(trainFilename+'.csv', usecols=['A Level Grade'])
yTrain
clf = clf.fit(xTrain, yTrain)

xTest = pd.read_csv(testFilename+'.csv', usecols=['SH1 CT', 'T3LT', 'Promo', 'SH2 LT (%)', 'SH2 CT (%)', 'Prelim Overall (%)'])
xTest
yPredict = clf.predict(xTest)                                         
print(yPredict)
pd.DataFrame(yPredict, columns=['Predicted A-level grades']).to_csv('Predicted A-level grades.csv')
