'''
Created on 20170424
@author: mao

这个是麦子学院里面的
'''
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

# open the csv file
allElectronicsData = open(r'test.csv', 'r')
reader = csv.reader(allElectronicsData)
# 字典读
dict_reader = csv.DictReader(allElectronicsData)
row = [row for row in dict_reader]

labelList = []

for i in row:
    # 去掉无用的字段
    del i["RID"]
    # 生成字典
    labelList.append(i.pop("class_buy_computer"))

# 特征列表里面是字典
featureList = row

# print(featureList)


vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:" + str(dummyX))

print(vec.get_feature_names())

print("labelList:" + str(labelList))
#
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:" + str(dummyY))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf:" + str(clf))
#
with open("test.dot", 'w')as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# predict
oneRowX = dummyX[0, :]
print("oneRowX:" + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[1] = 0
print("newRowX:" + str(newRowX))
predictedY = clf.predict(newRowX)
print("predictedY:" + str(predictedY))
