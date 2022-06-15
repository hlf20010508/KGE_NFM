################################
#This script provide the detailed complement of RF(baseline) and KGE_RF(discussed in paper 3.2.2)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import ampligraph as ampligraph
from ampligraph.datasets import load_from_csv
from ampligraph.evaluation import train_test_split_no_unseen,generate_corruptions_for_fit 
from ampligraph.latent_features import ComplEx,TransE,DistMult
import tensorflow as tf
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.utils import save_model,restore_model
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
#Des
################################################################
mms = MinMaxScaler(feature_range=(0,1))

fp_id = pd.read_csv('./data/yamanishi_08/791drug_struc.csv')['drug_id']
df_proseq = pd.read_csv('./data/yamanishi_08/989proseq.csv')
df_proseq.columns = ['pro_id','pro_ids','seq']
pro_id = df_proseq['pro_id']
drug_feats = np.loadtxt('./data/yamanishi_08/morganfp.txt',delimiter=',')
pro_feats = np.loadtxt('./data/yamanishi_08/pro_ctd.txt',delimiter=',')

pro_feats_scaled = mms.fit_transform(pro_feats)
pro_feats_scaled2 = PCA(n_components=100).fit_transform(pro_feats_scaled)
pro_feats_scaled3 = mms.fit_transform(pro_feats_scaled2)

fp_df = pd.concat([fp_id,pd.DataFrame(drug_feats)],axis=1)
prodes_df = pd.concat([pro_id,pd.DataFrame(pro_feats)],axis=1)

#Function
################################################################

def roc_auc(y, pred):
    fpr, tpr, threshold = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc


def pr_auc(y, pred):
    precision, recall, threshold = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return precision, recall, threshold, pr_auc


def get_scaled_embeddings(model,train_triples,test_triples,get_scaled,n_components):
    [train_sub_embeddings,test_sub_embeddings] = [model.get_embeddings(x['head'].values, embedding_type='entity') for x in [train_triples,test_triples]]
    [train_obj_embeddings,test_obj_embeddings] = [model.get_embeddings(x['tail'].values, embedding_type='entity') for x in [train_triples,test_triples]]
    train_feats = np.concatenate([train_sub_embeddings,train_obj_embeddings],axis=1)
    test_feats = np.concatenate([test_sub_embeddings,test_obj_embeddings],axis=1)
    train_dense_features = mms.fit_transform(train_feats)
    test_dense_features = mms.transform(test_feats)
    if get_scaled:
        pca = PCA(n_components=n_components)
        scaled_train_dense_features = pca.fit_transform(train_dense_features)
        scaled_pca_test_dense_features = pca.transform(test_dense_features)
    else:
        scaled_train_dense_features = train_dense_features
        scaled_pca_test_dense_features = test_dense_features
    return scaled_train_dense_features,scaled_pca_test_dense_features


def get_features(data,fp_df,prodes_df,use_pro):
    drug_features = pd.merge(data,fp_df,how='left',left_on='head',right_on='drug_id').iloc[:,4:1029].values
    pro_features = pd.merge(data,prodes_df,how='left',left_on='tail',right_on='pro_id').iloc[:,4:152].values
    if use_pro:
        feature = np.concatenate([drug_features,pro_features],axis=1)
    else:
        feature = drug_features
    return feature


def train(i):
    train = pd.read_csv('./data/yamanishi_08/data_folds/warm_start_1_10/train_fold_'+str(i+1)+'.csv')
    test = pd.read_csv('./data/yamanishi_08/data_folds/warm_start_1_10/test_fold_'+str(i+1)+'.csv')
    model = restore_model(model_name_path='output/kge_nfm/model/kge_nfm.pkl')   
    columns = ['head','relation','tail']
    # test_score = model.predict(test[columns])
    test_label = test['label'].values
    # #kge performance evaluation
    # roc_fpr, roc_tpr, roc_threshold, roc_a = roc_auc(test_label, test_score)
    # pr_precision, pr_recall, pr_threshold, pr_a = pr_auc(
    #     test_label, test_score)
    #pre
    re_train_all = train[columns]
    re_test_all = test[columns]
    train_dense_features,test_dense_features = get_scaled_embeddings(model,re_train_all,re_test_all,get_scaled=False,n_components=50)
    pca = PCA(n_components=500)
    train_dense_features_scaled = pca.fit_transform(train_dense_features)
    test_dense_features_scaled = pca.transform(test_dense_features)
    train_des = get_features(re_train_all,fp_df,prodes_df,use_pro=True)
    test_des = get_features(re_test_all,fp_df,prodes_df,use_pro=True)
    train_all_feats = np.concatenate([train_dense_features_scaled,train_des],axis=1)
    test_all_feats = np.concatenate([test_dense_features_scaled,test_des],axis=1)
    train_label = train['label']
    #rf
    clf = RandomForestClassifier(n_estimators=200,
                                criterion='entropy',
                                #max_depth=50,
                                random_state=0,
                                class_weight='balanced',
                                n_jobs=-1)
    clf.fit(train_des,train_label)
    pred = clf.predict_proba(test_des)
    roc_fpr_rf, roc_tpr_rf, roc_threshold_rf, roc_a_rf = roc_auc(test_label,pred[:,1])
    pr_precision_rf, pr_recall_rf, pr_threshold_rf, pr_a_rf = pr_auc(test_label,pred[:,1])

    #kge_rf
    clf = RandomForestClassifier(n_estimators=500,
                                criterion='entropy',
                                #max_depth=50,
                                random_state=0,
                                class_weight='balanced',
                                n_jobs=-1)
    clf.fit(train_all_feats,train_label)
    pred = clf.predict_proba(test_all_feats)
    roc_fpr_kge_rf, roc_tpr_kge_rf, roc_threshold_kge_rf, roc_a_kge_rf = roc_auc(test_label,pred[:,1])
    pr_precision_kge_rf, pr_recall_kge_rf, pr_threshold_kge_rf, pr_a_kge_rf = pr_auc(test_label,pred[:,1])

    return roc_fpr_rf, roc_tpr_rf, roc_threshold_rf, roc_a_rf, pr_precision_rf, pr_recall_rf, pr_threshold_rf, pr_a_rf, roc_fpr_kge_rf, roc_tpr_kge_rf, roc_threshold_kge_rf, roc_a_kge_rf, pr_precision_kge_rf, pr_recall_kge_rf, pr_threshold_kge_rf, pr_a_kge_rf

###train
ROC_AUC_RF = []
PR_AUC_RF = []
ROC_AUC_KGE_RF = []
PR_AUC_KGE_RF = []
for i in range(10):
    print(i)
    roc_fpr_rf, roc_tpr_rf, roc_threshold_rf, roc_a_rf, pr_precision_rf, pr_recall_rf, pr_threshold_rf, pr_a_rf, roc_fpr_kge_rf, roc_tpr_kge_rf, roc_threshold_kge_rf, roc_a_kge_rf, pr_precision_kge_rf, pr_recall_kge_rf, pr_threshold_kge_rf, pr_a_kge_rf = train(i)
    
    roc_curve_rf = pd.DataFrame()
    roc_curve_rf['roc_fpr_rf'] = roc_fpr_rf
    roc_curve_rf['roc_tpr_rf'] = roc_tpr_rf
    roc_curve_rf['roc_threshold_rf'] = roc_threshold_rf
    print(roc_curve_rf)

    pr_curve_rf = pd.DataFrame()
    pr_curve_rf['pr_precision_rf'] = pr_precision_rf
    pr_curve_rf['pr_recall_rf'] = pr_recall_rf
    pr_curve_rf['pr_threshold_rf'] = np.append(pr_threshold_rf, values=np.nan)

    print(pr_curve_rf)

    roc_curve_kge_rf = pd.DataFrame()
    roc_curve_kge_rf['roc_fpr_kge_rf'] = roc_fpr_kge_rf
    roc_curve_kge_rf['roc_tpr_kge_rf'] = roc_tpr_kge_rf
    roc_curve_kge_rf['roc_threshold_kge_rf'] = roc_threshold_kge_rf
    print(roc_curve_kge_rf)

    pr_curve_kge_rf = pd.DataFrame()
    pr_curve_kge_rf['pr_precision_kge_rf'] = pr_precision_kge_rf
    pr_curve_kge_rf['pr_recall_kge_rf'] = pr_recall_kge_rf
    pr_curve_kge_rf['pr_threshold_kge_rf'] = np.append(
        pr_threshold_kge_rf, values=np.nan)

    print(pr_curve_kge_rf)

    roc_curve_rf.to_csv('output/kge_rf/curve/roc_rf/'+str(i)+'.csv')
    pr_curve_rf.to_csv('output/kge_rf/curve/pr_rf/'+str(i)+'.csv')
    roc_curve_kge_rf.to_csv('output/kge_rf/curve/roc_kge_rf/'+str(i)+'.csv')
    pr_curve_kge_rf.to_csv('output/kge_rf/curve/pr_kge_rf/'+str(i)+'.csv')

    print('roc_auc_rf: %f' % roc_a_rf)
    print('pr_auc_rf: %f' % pr_a_rf)
    print('roc_auc_kge_rf: %f' % roc_a_kge_rf)
    print('pr_auc_kge_rf: %f' % pr_a_kge_rf)
    
    ROC_AUC_RF.append(roc_a_rf)
    PR_AUC_RF.append(pr_a_rf)
    ROC_AUC_KGE_RF.append(roc_a_kge_rf)
    PR_AUC_KGE_RF.append(pr_a_kge_rf)

stable_metrics = pd.DataFrame()
stable_metrics['roc_auc_rf'] = ROC_AUC_RF
stable_metrics['pr_auc_rf'] = PR_AUC_RF
stable_metrics['roc_auc_kge_rf'] = ROC_AUC_KGE_RF
stable_metrics['pr_auc_kge_rf'] = PR_AUC_KGE_RF
print(stable_metrics)
print(stable_metrics.describe())

stable_metrics.to_csv('output/kge_rf/auc/kge_rf_auc.csv')
