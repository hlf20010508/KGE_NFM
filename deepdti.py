################################
#This script provide a demo of MPNN_CNN & DeepDTI, the runtime on one fold mainly takes 3~5 hours (V100). 

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import metrics
import numpy as np 
from sklearn.model_selection import train_test_split
from datetime import datetime
#Load Data
################################################################
dt_08 = pd.read_csv('./data/yamanishi_08/dt_all_08.txt',delimiter='\t',header=None)
dt_08.columns = ['head','relation','tail']

df_drug = pd.read_csv('./data/yamanishi_08/791drug_struc.csv')
df_proseq = pd.read_csv('./data/yamanishi_08/989proseq.csv')
df_proseq.columns = ['pro_id','pro_ids','seq']

pro_id = df_proseq['pro_id']

#define function
################################

def get_struc(data,df_drug,df_proseq):
    drug_struc = pd.merge(data,df_drug,how='left',left_on='head',right_on='drug_id')['smiles'].values
    pro_struc = pd.merge(data,df_proseq,how='left',left_on='tail',right_on='pro_id')['seq'].values
    return drug_struc,pro_struc


def roc_auc(y,pred):
    fpr, tpr, threshold = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc

def pr_auc(y, pred):
    precision, recall, threshold = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return precision, recall, threshold, pr_auc


data_path = './data/yamanishi_08/data_folds/warm_start_1_10/'  
def load_data(i):
    train = pd.read_csv(data_path+'train_fold_'+str(i+1)+'.csv')
    test = pd.read_csv(data_path+'test_fold_'+str(i+1)+'.csv')
    return train,test

#need to be adjusted when changing methods
drug_encoding, target_encoding = 'MPNN', 'CNN'

def get_input(train_all,test_all):
    train_label = train_all['label']
    test_label = test_all['label']
    train_re, valid_re, y_train, y_valid = train_test_split(train_all[['head','relation','tail']],train_label,test_size=0.01, 
                                                                random_state=0,
                                                                stratify=train_label)
    train_drug_feats,train_pro_feats = get_struc(train_re,df_drug,df_proseq)
    valid_drug_feats,valid_pro_feats = get_struc(valid_re,df_drug,df_proseq)
    test_drug_feats,test_pro_feats = get_struc(test_all,df_drug,df_proseq)
    train = utils.data_process(train_drug_feats, train_pro_feats, y_train, 
                                drug_encoding, target_encoding, 
                                split_method='no_split',
                                random_seed = 0)
    valid = utils.data_process(valid_drug_feats, valid_pro_feats, y_valid, 
                            drug_encoding, target_encoding, 
                            split_method='no_split',
                            random_seed = 0)                            
    test = utils.data_process(test_drug_feats, test_pro_feats, test_label, 
                                drug_encoding, target_encoding, 
                                split_method='no_split',
                                random_seed = 0)
    return train,valid,test


######################################## Training
#parameters for MPNN_CNN
config = utils.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                        cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 20, 
                         LR = 0.001, 
                         batch_size = 256,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,64],
                         cnn_target_kernels = [4,8,8]
                        )

'''
#parameters for DeepDTI
config = utils.generate_config(drug_encoding, target_encoding, 
                            cls_hidden_dims = [1024,1024,512],
                            train_epoch = 300, 
                            LR = 0.001, 
                            batch_size = 5000, 
                            cnn_drug_filters = [32,64,96],
                            cnn_drug_kernels = [4,8,12], 
                            cnn_target_filters = [32,64,96], 
                            cnn_target_kernels = [4,8,12])
'''
ROC_AUC = []
PR_AUC = []
for i in range(10):
    print(i)
    train,test = load_data(i)
    train_input,valid_input,test_input = get_input(train,test)
    # model=models.model_pretrained('runs\model\9')
    model = models.model_initialize(**config)
    model.train(train_input,valid_input)

    test_score=model.predict(test_input)
    test_label=test['label'].values

    roc_fpr, roc_tpr, roc_threshold, roc_a = roc_auc(test_label,test_score)
    pr_precision, pr_recall, pr_threshold, pr_a = pr_auc(test_label,test_score)

    roc_curve = pd.DataFrame()
    roc_curve['roc_fpr'] = roc_fpr
    roc_curve['roc_tpr'] = roc_tpr
    roc_curve['roc_threshold'] = roc_threshold
    print(roc_curve)

    pr_curve = pd.DataFrame()
    pr_curve['pr_precision'] = pr_precision
    pr_curve['pr_recall'] = pr_recall
    pr_curve['pr_threshold'] = np.append(pr_threshold,values=np.nan) #

    print(pr_curve)

    roc_curve.to_csv('output/deepdti/curve/roc/'+str(i)+'.csv')
    pr_curve.to_csv('output/deepdti/curve/pr/'+str(i)+'.csv')

    print('roc_auc: %f'%roc_a)
    print('pr_auc: %f'%pr_a)

    ROC_AUC.append(roc_a)
    PR_AUC.append(pr_a)

    model.save_model('output/deepdti/model/'+str(i))

stable_metrics = pd.DataFrame()
stable_metrics['roc_auc'] = ROC_AUC
stable_metrics['pr_auc'] = PR_AUC
print(stable_metrics)
print(stable_metrics.describe())

stable_metrics.to_csv('output/deepdti/auc/deepdti_auc.csv')
