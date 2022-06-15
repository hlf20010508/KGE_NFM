################################
# This script provide a demo of KGE_NFM & NFM, the runtime on one fold mainly takes 40~50 minutes.

from tensorflow.python.keras.optimizers import Adam, Adagrad, Adamax
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from ampligraph.latent_features import ComplEx, TransE, DistMult
from ampligraph.evaluation import train_test_split_no_unseen, generate_corruptions_for_fit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ampligraph.utils import save_model, restore_model
from datetime import datetime
from tensorflow import keras
from sklearn.decomposition import PCA
from tensorflow.python.keras.callbacks import EarlyStopping
from deepctr.models import NFM
from sklearn import metrics
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.evaluation import evaluate_performance
import tensorflow as tf
from ampligraph.datasets import load_from_csv
import ampligraph as ampligraph
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load data
################################################################
# data example: yamanishi_08
dt_08 = pd.read_csv('./data/yamanishi_08/dt_all_08.txt',
                    delimiter='\t', header=None)
dt_08.columns = ['head', 'relation', 'tail']

# kg
kg1 = pd.read_csv('./data/yamanishi_08/kg_data/kegg_kg.txt',
                  delimiter='\t', header=None)
kg2 = pd.read_csv(
    './data/yamanishi_08/kg_data/yamanishi_uniprot_kg.txt', delimiter='\t', header=None)
kg = pd.concat([kg1, kg2])
kg.index = range(len(kg))
kg.columns = ['head', 'relation', 'tail']

# for nfm input
head_le = LabelEncoder()
tail_le = LabelEncoder()
head_le.fit(dt_08['head'].values)
tail_le.fit(dt_08['tail'].values)

mms = MinMaxScaler(feature_range=(0, 1))

# descriptors preparation
fp_id = pd.read_csv('./data/yamanishi_08/791drug_struc.csv')['drug_id']
df_proseq = pd.read_csv('./data/yamanishi_08/989proseq.csv')
df_proseq.columns = ['pro_id', 'pro_ids', 'seq']
pro_id = df_proseq['pro_id']
drug_feats = np.loadtxt('./data/yamanishi_08/morganfp.txt', delimiter=',')
pro_feats = np.loadtxt('./data/yamanishi_08/pro_ctd.txt', delimiter=',')

pro_feats_scaled = mms.fit_transform(pro_feats)
pro_feats_scaled2 = PCA(n_components=100).fit_transform(pro_feats_scaled)
pro_feats_scaled3 = mms.fit_transform(pro_feats_scaled2)

fp_df = pd.concat([fp_id, pd.DataFrame(drug_feats)], axis=1)
prodes_df = pd.concat([pro_id, pd.DataFrame(pro_feats_scaled3)], axis=1)

# Function
################################################################

# If you want to test other scenarios, just change the data path.
# But it should be noted that the hypermeters in nfm need to be adjusted.
# Typiclly, the l2_reg_dnn & l2_reg_linear = 1e-5 is enough in the warm start.
# For the cold start, the l2_reg_dnn & l2_reg_linear need to be larger, like 1e-3.

data_path = './data/yamanishi_08/data_folds/warm_start_1_10/'


def load_data(i):
    train = pd.read_csv(data_path+'train_fold_'+str(i+1) +
                        '.csv')[['head', 'relation', 'tail', 'label']]
    train_pos = train[train['label'] == 1]
    test = pd.read_csv(data_path+'test_fold_'+str(i+1) +
                       '.csv')[['head', 'relation', 'tail', 'label']]
    data = pd.concat([train_pos, kg])[['head', 'relation', 'tail']]
    return train, train_pos, test, data


def roc_auc(y, pred):
    fpr, tpr, threshold = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc


def pr_auc(y, pred):
    precision, recall, threshold = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return precision, recall, threshold, pr_auc


def get_scaled_embeddings(model, train_triples, test_triples, get_scaled, n_components):
    [train_sub_embeddings, test_sub_embeddings] = [model.get_embeddings(
        x['head'].values, embedding_type='entity') for x in [train_triples, test_triples]]
    [train_obj_embeddings, test_obj_embeddings] = [model.get_embeddings(
        x['tail'].values, embedding_type='entity') for x in [train_triples, test_triples]]
    train_feats = np.concatenate(
        [train_sub_embeddings, train_obj_embeddings], axis=1)
    test_feats = np.concatenate(
        [test_sub_embeddings, test_obj_embeddings], axis=1)
    train_dense_features = mms.fit_transform(train_feats)
    test_dense_features = mms.transform(test_feats)
    if get_scaled:
        pca = PCA(n_components=n_components)
        scaled_train_dense_features = pca.fit_transform(train_dense_features)
        scaled_pca_test_dense_features = pca.transform(test_dense_features)
    else:
        scaled_train_dense_features = train_dense_features
        scaled_pca_test_dense_features = test_dense_features
    return scaled_train_dense_features, scaled_pca_test_dense_features


def get_features(data, fp_df, prodes_df, use_pro):
    drug_features = pd.merge(
        data, fp_df, how='left', left_on='head', right_on='drug_id').iloc[:, 4:1029].values
    pro_features = pd.merge(data, prodes_df, how='left',
                            left_on='tail', right_on='pro_id').iloc[:, 4:105].values
    if use_pro:
        feature = np.concatenate([drug_features, pro_features], axis=1)
    else:
        feature = drug_features
    return feature

# 'DenseFeat("des",train_des.shape[1]),'des':train_des,' is used for nfm training


def get_nfm_input(re_train_all, re_test_all, train_feats, test_feats, train_des, test_des, embedding_dim, pca_components):
    train_all_feats = np.concatenate([train_feats, train_des], axis=1)
    test_all_feats = np.concatenate([test_feats, test_des], axis=1)
    train_all_feats_scaled = mms.fit_transform(train_all_feats)
    test_all_feats_scaled = mms.transform(test_all_feats)
    feature_columns = [SparseFeat('head', re_train_all['head'].unique().shape[0], embedding_dim=embedding_dim),
                       SparseFeat('tail', re_train_all['tail'].unique(
                       ).shape[0], embedding_dim=embedding_dim),
                       DenseFeat("feats", train_all_feats_scaled.shape[1]),
                       # DenseFeat("des",train_des.shape[1])
                       ]
    train_model_input = {'head': head_le.transform(re_train_all['head'].values),
                         'tail': tail_le.transform(re_train_all['tail'].values),
                         'feats': train_all_feats_scaled,
                         # 'des':train_des
                         }
    test_model_input = {'head': head_le.transform(re_test_all['head'].values),
                        'tail': tail_le.transform(re_test_all['tail'].values),
                        'feats': test_all_feats_scaled,
                        # 'des':test_des
                        }
    return feature_columns, train_model_input, test_model_input

# the hypermeters(l2_reg_dnn & l2_reg_linear) need to be adjusted in cold start scenarios, like 1e-3


def train_nfm(feature_columns, train_model_input, train_label, test_model_input, y, patience):
    re_model = NFM(feature_columns, feature_columns, task='binary', dnn_hidden_units=(128, 128),
                   l2_reg_dnn=1e-5, l2_reg_linear=1e-5,
                   )
    re_model.compile(Adam(1e-3), "binary_crossentropy",
                     metrics=[keras.metrics.Precision(name='precision'), ], )
    es = EarlyStopping(monitor='loss', patience=patience,
                       min_delta=0.0001, mode='min', restore_best_weights=True)
    history = re_model.fit(train_model_input, train_label,
                           batch_size=20000, epochs=2000,
                           verbose=2,
                           callbacks=[es]
                           )
    pred_y = re_model.predict(test_model_input, batch_size=512)

    roc_fpr, roc_tpr, roc_threshold, roc_a = roc_auc(y, pred_y[:, 0])
    pr_precision, pr_recall, pr_threshold, pr_a = pr_auc(y, pred_y[:, 0])

    return roc_fpr, roc_tpr, roc_threshold, roc_a, pr_precision, pr_recall, pr_threshold, pr_a, pred_y[:, 0]


def train(i, test_num_neg, train_num_neg, embedding_dim, n_components, use_pro, patience):
    train, train_pos, test, data = load_data(i)
    model = DistMult(batches_count=10000,
                     seed=0,
                     epochs=50,
                     k=400,
                     # embedding_model_params={'corrupt_sides':'o'},
                     optimizer='adam',
                     optimizer_params={'lr': 1e-3},
                     loss='pairwise',  # pairwise
                     regularizer='LP',
                     regularizer_params={'p': 3, 'lambda': 1e-5},
                     verbose=True)
    model.fit(data.values, early_stopping=True, early_stopping_params={
        # validation set, here we use training set for validation
        'x_valid': train_pos[['head', 'relation', 'tail']].values,
        'criteria': 'mrr',         # Uses mrr criteria for early stopping
        'burn_in': 10,              # early stopping kicks in after 10 epochs
        'check_interval': 2,         # validates every 2th epoch
        # stops if 3 successive validation checks are bad.
                    'stop_interval': 3,
                    'x_filter': dt_08.values,          # Use filter for filtering out positives
                    # corrupt object (but not at once)
                    'corrupt_side': 'o'
    })
    save_model(model, model_name_path='output/kge_nfm/model/kge_nfm.pkl')
    # model = restore_model(
    #     model_name_path='./my_dismult_400_warm_1_10.pkl')
    columns = ['head', 'relation', 'tail']
    test_score = model.predict(test[columns])
    test_label = test['label'].values

    roc_fpr, roc_tpr, roc_threshold, roc_a = roc_auc(test_label, test_score)
    pr_precision, pr_recall, pr_threshold, pr_a = pr_auc(
        test_label, test_score)

    # nfm preparation
    re_train_all = train[columns]
    re_test_all = test[columns]
    train_label = train['label']
    train_dense_features, test_dense_features = get_scaled_embeddings(
        model, re_train_all, re_test_all, False, n_components)
    train_des = get_features(re_train_all, fp_df, prodes_df, use_pro)
    test_des = get_features(re_test_all, fp_df, prodes_df, use_pro)
    feature_columns, train_model_input, test_model_input = get_nfm_input(re_train_all, re_test_all,
                                                                         train_dense_features, test_dense_features,
                                                                         train_des, test_des,
                                                                         embedding_dim, n_components)
    roc_fpr_nfm, roc_tpr_nfm, roc_threshold_nfm, roc_a_nfm, pr_precision_nfm, pr_recall_nfm, pr_threshold_nfm, pr_a_nfm, pred_y = train_nfm(
        feature_columns, train_model_input, train_label, test_model_input, test_label, patience)
    return roc_fpr, roc_tpr, roc_threshold, roc_a, pr_precision, pr_recall, pr_threshold, pr_a, roc_fpr_nfm, roc_tpr_nfm, roc_threshold_nfm, roc_a_nfm, pr_precision_nfm, pr_recall_nfm, pr_threshold_nfm, pr_a_nfm, re_train_all, train_label, re_test_all, test_label, pred_y


#train and test
# the early stopping parameter in nfm, referring patience, need to be adjusted in cold start scenarios, like 15~20
################################################################
ROC_AUC = []
PR_AUC = []
ROC_AUC_NFM = []
PR_AUC_NFM = []
for i in range(10):
    print(i)
    roc_fpr, roc_tpr, roc_threshold, roc_a, pr_precision, pr_recall, pr_threshold, pr_a, roc_fpr_nfm, roc_tpr_nfm, roc_threshold_nfm, roc_a_nfm, pr_precision_nfm, pr_recall_nfm, pr_threshold_nfm, pr_a_nfm, re_train_all, train_label, re_test_all, test_label, pred_y = train(
        i, 10, 10, 50, 200, True, 10)

    re_train_all['label'] = train_label
    re_test_all['label'] = test_label
    re_test_all['pred'] = pred_y

    roc_curve = pd.DataFrame()
    roc_curve['roc_fpr'] = roc_fpr
    roc_curve['roc_tpr'] = roc_tpr
    roc_curve['roc_threshold'] = roc_threshold
    print(roc_curve)

    pr_curve = pd.DataFrame()
    pr_curve['pr_precision'] = pr_precision
    pr_curve['pr_recall'] = pr_recall
    pr_curve['pr_threshold'] = np.append(pr_threshold, values=np.nan)

    print(pr_curve)

    roc_curve_nfm = pd.DataFrame()
    roc_curve_nfm['roc_fpr_nfm'] = roc_fpr_nfm
    roc_curve_nfm['roc_tpr_nfm'] = roc_tpr_nfm
    roc_curve_nfm['roc_threshold_nfm'] = roc_threshold_nfm
    print(roc_curve_nfm)

    pr_curve_nfm = pd.DataFrame()
    pr_curve_nfm['pr_precision_nfm'] = pr_precision_nfm
    pr_curve_nfm['pr_recall_nfm'] = pr_recall_nfm
    pr_curve_nfm['pr_threshold_nfm'] = np.append(
        pr_threshold_nfm, values=np.nan)

    print(pr_curve_nfm)

    roc_curve.to_csv('output/kge_nfm/curve/roc/'+str(i)+'.csv')
    pr_curve.to_csv('output/kge_nfm/curve/pr/'+str(i)+'.csv')
    roc_curve_nfm.to_csv('output/kge_nfm/curve/roc_nfm/'+str(i)+'.csv')
    pr_curve_nfm.to_csv('output/kge_nfm/curve/pr_nfm/'+str(i)+'.csv')

    print('roc_auc: %f' % roc_a)
    print('pr_auc: %f' % pr_a)
    print('roc_auc_nfm: %f' % roc_a_nfm)
    print('pr_auc_nfm: %f' % pr_a_nfm)

    ROC_AUC.append(roc_a)
    PR_AUC.append(pr_a)
    ROC_AUC_NFM.append(roc_a_nfm)
    PR_AUC_NFM.append(pr_a_nfm)


stable_metrics = pd.DataFrame()
stable_metrics['roc_auc'] = ROC_AUC
stable_metrics['pr_auc'] = PR_AUC
stable_metrics['roc_auc_nfm'] = ROC_AUC_NFM
stable_metrics['pr_auc_nfm'] = PR_AUC_NFM
print(stable_metrics)
print(stable_metrics.describe())

stable_metrics.to_csv('output/kge_nfm/auc/kge_nfm_auc.csv')
