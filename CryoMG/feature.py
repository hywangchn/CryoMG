import re
import numpy as np
import pandas as pd
import pickle
import gc
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from featureSelection import ttest, Ttest_RFE_SVM_loop, TRankAcc
from culAcc import SKFold_accuracy_cross_Grid
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import make_scorer, fbeta_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate,train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix

# path1 = r"E:/code/GryoGM/features/test/images_HESSIAN_test_0.csv"
# path2 = r"E:/code/GryoGM/features/test/images_HESSIAN_test_1.csv"
# path3 = r"E:/code/GryoGM/features/val/images_HESSIAN_val_0.csv"
# path4 = r"E:/code/GryoGM/features/val/images_HESSIAN_val_1.csv"

# data1 = pd.read_csv(path1, sep = ',', index_col=0)
# data2 = pd.read_csv(path2, sep = ',', index_col=0)
# data3 = pd.read_csv(path3, sep = ',', index_col=0)
# data4 = pd.read_csv(path4, sep = ',', index_col=0)
# print('==========load data finish==========')

# res = pd.concat([data1,data2,data3,data4], axis=0)
# print('==========concat data finish==========')

# data1 = None
# data2 = None
# data3 = None
# data4 = None
# gc.collect()

# res = (res-res.mean())/(res.max()-res.min())
# print('==========cul res finish==========')

# res_var = res.var(axis=0)
# print('==========cul var finish==========')

# col = []
# for i in range(len(res_var)):
#     if res_var[i]>0.05:
#         col.append(str(i))

# result = res[:][col].round(4)
# result.to_csv('./features/HESSIAN_round4.csv')
#################################################################

# path1 = r'E:\code\GryoGM\fea\resnet152_new_feature_val.csv'
# path2 = r'E:\code\GryoGM\fea\resnet152_new_feature_test.csv'
#
# data1 = pd.read_csv(path1, sep = ',', index_col=0)
# data2 = pd.read_csv(path2, sep = ',', index_col=0)
# print('==========load data finish==========')
#
# res = pd.concat([data1,data2], axis=0)
# print('==========concat data finish==========')
#
# res = (res-res.mean())/(res.max()-res.min())
# print('==========cul res finish==========')
#
# # res_var = res.var(axis=0)
# # print('==========cul var finish==========')
#
# # col = []
# # for i in range(len(res_var)):
# #     if res_var[i]>0.05:
# #         col.append(str(i))
#
# # result = res[:][col].round(4)
# res.index = res.index+".jpg"
# res.to_csv('./features/resnet152_round4.csv')

###############################################################

# path = r'./features/data'
# result = None

# lll = pd.read_csv('./features/label.csv', sep = ',', index_col=0)

# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file[-3:]=='csv':
#             print(file)
#             temp = pd.read_csv(os.path.join(root, file), sep=',', index_col=0)
#             temp.columns = file[:-11] + temp.columns
#             ############################################################################
#             label = []
#             for i in temp.index:
#                 label.append(lll.loc[i]["label"])

#             features_ttest_index, _ = ttest(temp.values, label)
#             temp = temp[temp.columns[features_ttest_index[:30]]]
#             ############################################################################
#             if result is None:
#                 result = temp
#             else:
#                 result = pd.concat([result,temp], axis=1)
#                 temp = None
# result.to_csv('./features/data_all/features.csv')

################################################

# path = './features/features.csv'
# data = pd.read_csv(path, sep = ',', index_col=0)
# path_label = './data/'
# lll = pd.DataFrame(columns=(['label']))
# for root, dirs, files in os.walk(path_label):
#     for file in files:
#         lll.loc[file] = root[-1]
#
# label = pd.DataFrame(columns=(['label']))
# for i in data.index:
#     label.loc[i] = lll.loc[i]
#
# label.to_csv('./features/label.csv')

######################################################

# matplotlib.use("Qt5Agg")

path = './features'
data = pd.read_csv(os.path.join(path,'data_all/features.csv'), sep = ',', index_col=0)
label = pd.read_csv(os.path.join(path,'label.csv'), sep = ',', index_col=0)

print("=================load data finish======================")

col = [i for i in data.columns if i[:3]!="LBP"]
data = data.loc[:][col]
data_var = data.var(axis=0)

col = []
for i in range(len(data_var)):
    if data_var[i]>0.02:
        col.append(data.columns[i])
        
data2 = data.loc[:][col]

print("=================var data finish======================")

corr_matrix = data2.corr()

print("=================corr data finish======================")

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
# plt.title("Correlation Matrix")
plt.savefig("./CorrelationMatrix.tif",dpi=600)
plt.show()

####################################################################
# path = './features/data_all/'
# lll = pd.read_csv('./features/label.csv', sep = ',', index_col=0)
# feanum_record = {}
# for root, dirs, files in os.walk(path):
#     for file in files:
#         features = pd.read_csv(os.path.join(path,file), sep = ',', index_col=0)
#         label = []

#         for i in features.index:
#             label.append(lll.loc[i]["label"])
#         features_col = features.columns
#         features = features.values
#         matrix_record_dict = {}
#         pre_label_record_dict = {}
#         for feature_number in range(1,31):
#             clf, feanum, max_score_c, max_matrix_c, pre_max = Ttest_RFE_SVM_loop(features, np.array(label), feature_number)
#             res, pre_label = SKFold_accuracy_cross_Grid(features[:,feanum],label)
#             res["svm"] = max_matrix_c
#             pre_label['svm'] = pre_max
#             matrix_record_dict[str(feature_number)] = res
#             pre_label_record_dict[str(feature_number)] = pre_label
#             print("===========================================================")
#             print("current feature name : "+file)
#             print("current feature number : "+str(feature_number))
#             print("===========================================================")
#             feanum_record[str(feature_number)] = feanum

#         res = pd.DataFrame(columns=range(1, 31))
#         for i in feanum_record.keys():
#             temp = []
#             temp = temp + features_col[feanum_record[i]].tolist()
#             for j in range(len(temp), 30):
#                 temp.append(0)
#             res.loc[i] = temp
#         res.to_excel(r'E:\code\GryoGM\features_col.xlsx')


#         with open(('./result/'+file[:-4]+'.pkl'), 'wb') as f:
#             pickle.dump(matrix_record_dict, f)

#         with open(('./result/'+file[:-4]+'pre_label.pkl'), 'wb') as f:
#             pickle.dump(pre_label_record_dict, f)

##########################################################################

# metric = ['accuracy','precision', 'recall', 'f1_score']

# path = r'./result/'
# method = ['svm','bayes','dtree','xgb','lgb','knn']

# # for i in method:
# #     vars()['res_'+i] = pd.DataFrame(columns=metric)

# # res_svm = pd.DataFrame(columns=metric)
# # res_bayes = pd.DataFrame(columns=metric)
# # res_dtree = pd.DataFrame(columns=metric)
# # res_xgb = pd.DataFrame(columns=metric)
# # res_lgb = pd.DataFrame(columns=metric)
# # res_knn = pd.DataFrame(columns=metric)

# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file[-10:]!="round4.pkl":
#         # if file!="features.pkl":
#             continue
#         for i in method:
#             vars()['res_' + i] = pd.DataFrame(columns=metric)

#         with open(os.path.join(root,file), 'rb') as f:
#             data = pickle.load(f)

#         for k in data.keys():
#             for i in data[k].keys():
#                 matrix = np.sum(data[k][i],axis=0)/10
#                 acc = ((matrix[0]+matrix[3])/(matrix[0]+matrix[1]+matrix[2]+matrix[3]))
#                 precision = matrix[3]/(matrix[1]+matrix[3])
#                 recall = matrix[3]/(matrix[2]+matrix[3])
#                 f1 = 2/(1/precision+1/recall)
#                 vars()['res_' + i].loc[k] = [acc, precision, recall, f1]
#                 # if vars()['res_' + i].loc[k].isnull().any():
#                 #     print(matrix)

#         record = pd.DataFrame(index=[str(i) for i in range(1,31)])
#         for i in method:
#             vars()['res_' + i].columns = i+"_"+vars()['res_' + i].columns
#             vars()['res_' + i].fillna(0, inplace=True)
#             record = pd.concat([record, vars()['res_' + i]], axis=1)
#         record = record*100
#         record = record.round(3)
#         record.to_excel(os.path.join(root,file[:-4]+".xlsx"))
#         print(record)


###########################################################################

# lll = pd.read_csv('./features/label.csv', sep = ',', index_col=0)

# with open(r"./result/swinTransformer_round4pre_label.pkl", "rb") as f:
#     data = pickle.load(f)

# for i in data.keys():
#     for k in data[i].keys():
#         data[i][k] = data[i][k][np.lexsort((data[i][k][:,1],data[i][k][:,0])),:]
# # lll_list = lll.index.tolist()

# # for i in data.keys():
# #     print(i)
# #     for j in data[i].keys():
# #         print(j)

# path = "./data/GryoGM"

# for dirsss in os.listdir(path):
#     vars()["pre_label_"+dirsss] = []
#     for file in os.listdir(os.path.join(path,dirsss,"micrographs")):
#         if file in lll.index:
#             vars()["pre_label_"+dirsss].append(lll.index.get_loc(file))
#             if lll.loc[file]["label"]==0:
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"HorizontalFlip.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate135.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate180.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate225.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate270.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate315.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate45.jpg"))
#                 vars()["pre_label_"+dirsss].append(lll.index.get_loc(file[:-4]+"rotate90.jpg"))

# # length = 0
# # for dirsss in os.listdir(path):
# #     length+=len(vars()["pre_label_"+dirsss])
# # print(length)
# method = ["svm","bayes","dtree","xgb","lgb"]
# res = ["acc","precision","recall","f1"]
# col = []
# for i in method:
#     for j in res:
#         col.append(i+"_"+j)
# for dirsss in os.listdir(path):
#     record = pd.DataFrame(columns=col)
#     for kkk in data.keys():
#         temp = []
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1),data[kkk]["svm"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["bayes"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["dtree"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["xgb"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["lgb"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         record.loc[kkk] = temp
#     record.to_excel("./every/"+dirsss+".xlsx")

###########################################################################

# result = pd.DataFrame(columns=(["acc","precision","recall","f1"]))
# path = r"E:\code\mmpretrain-main\tools\work_dirs"


# for dirsss in os.listdir(path):
#     for root, dirs, files in os.walk(os.path.join(path,dirsss)):
#         for file in files:
#             if file[-4:]==".log":
#                 with open(os.path.join(root,file), "rb") as f:
#                     data = f.readlines()
#                     matrix = np.array([
#                         [int(re.compile(".([0-9]+),").findall(str(data[-3]))[0]),
#                           int(re.compile(".([0-9]+)]").findall(str(data[-3]))[0])
#                           ],
#                         [int(re.compile(".([0-9]+),").findall(str(data[-2]))[0]),
#                           int(re.compile(".([0-9]+)]").findall(str(data[-2]))[0])
#                           ]
#                         ])
#                     matrix = matrix.ravel()
                    
#                     acc = ((matrix[0]+matrix[3])/(matrix[0]+matrix[1]+matrix[2]+matrix[3]))
#                     precision = matrix[3]/(matrix[1]+matrix[3])
#                     recall = matrix[3]/(matrix[2]+matrix[3])
#                     f1 = 2/(1/precision+1/recall)
                    
#                     ind = re.compile("diy_([^-]*)[-]").findall(dirsss)[0]
#                     result.loc[ind] = [acc,precision,recall,f1]
# result = result*100
# result = result.round(4)
# result.to_excel('./deeplearning_acc.xlsx')

###########################################################################

# path = r"E:\code\GryoGM\every"

# num = {
#         "CANNY":2,
#         "GLCM":16,
#         "HESSIAN":3,
#         "HOG":30,
#         "LBP":30,
#         "resnet152":29,
#         "swinTransformer":22,
#         "features":28
#         }
# record = {}

# for dirsss in os.listdir(path):
#     record_temp = {}
#     ind = num[dirsss]-1
#     for file in os.listdir(os.path.join(path,dirsss)):
#         data = pd.read_excel(os.path.join(path,dirsss,file), index_col=0).fillna(1)
#         record_temp[file] = data.iloc[ind]["svm_acc"]
#     record[dirsss] = record_temp
# pd.DataFrame(record).to_excel('./dandufenxi.xlsx')

###########################################################################

# lll = pd.read_csv('./features/label.csv', sep = ',', index_col=0)

# with open(r"./result/featurespre_label.pkl", "rb") as f:
#     data = pickle.load(f)

# for i in data.keys():
#     for k in data[i].keys():
#         data[i][k] = data[i][k][np.lexsort((data[i][k][:,1],data[i][k][:,0])),:]
# # lll_list = lll.index.tolist()

# # for i in data.keys():
# #     print(i)
# #     for j in data[i].keys():
# #         print(j)

# path = "./data/GryoGM"

# for dirsss in os.listdir(path):
#     vars()["pre_label_"+dirsss] = []
#     for file in os.listdir(os.path.join(path,dirsss,"micrographs")):
#         if file in lll.index:
#             vars()["pre_label_"+dirsss].append(lll.index.get_loc(file))

# # length = 0
# # for dirsss in os.listdir(path):
# #     length+=len(vars()["pre_label_"+dirsss])
# # print(length)
# method = ["svm","bayes","dtree","xgb","lgb"]
# res = ["acc","precision","recall","f1"]
# col = []
# for i in method:
#     for j in res:
#         col.append(i+"_"+j)
# for dirsss in os.listdir(path):
#     record = pd.DataFrame(columns=col)
#     for kkk in data.keys():
#         temp = []
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1),data[kkk]["svm"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["bayes"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["dtree"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["xgb"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         matrix = confusion_matrix(lll.iloc[vars()["pre_label_"+dirsss]].values.reshape(-1,),data[kkk]["lgb"][vars()["pre_label_"+dirsss],1], labels=[0,1]).ravel()
#         acc = ((matrix[0]/(matrix[0]+matrix[1])+matrix[3]/(matrix[2]+matrix[3])))/2
#         precision = matrix[3]/(matrix[1]+matrix[3])
#         recall = matrix[3]/(matrix[2]+matrix[3])
#         f1 = 2/(1/precision+1/recall)
#         temp+=[acc,precision,recall,f1]
        
#         record.loc[kkk] = temp
#     record.to_excel("./Enhance/noEnhance/"+dirsss+".xlsx")

# record = {}
# for file in os.listdir("./Enhance/features"):
#     dataEnhance = pd.read_excel(os.path.join("./Enhance/features",file))
#     dataNoEnhance = pd.read_excel(os.path.join("./Enhance/noEnhance",file))
#     record[file] = [max(dataEnhance["svm_acc"]),max(dataNoEnhance["svm_acc"])]
# record = pd.DataFrame(record).fillna(1)*100
# record.to_excel("./Enhance/Enhance.xlsx")























