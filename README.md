## A unified DTI prediction framework based on knowledge graph and recommendation system

# Code and data description
## Scripts
- `kge_nfm.py`: the complement of the KGE_NFM & NFM methods.
- `kge_rf.py`: the complement of the KGE_RF & RF methods.
- `deepdit.py`: the complement of the MPNN_CNN & DeepDTI methods.
- the complement of DTINet and DTiGEMS is tested based on their source packages (more in Prerequisites)


## `data/` directory
#### `yamanishi_08/` directory
- `data_folds/`: 10 folds training set and test set in the three scenarios
    - `warm_start_1_1/` 
    - `warm_start_1_10/` 
    - `drug_coldstart/` 
    - `protein_coldstart/` 
- `kg_data/`: supporting knowledge graph data
- `dt_all_08.csv`: whole DTI dataset
- `791drug_struc.csv`: drugbank id and smiles of drugs
- `989proseq.csv`: kegg id and sequences of proteins 
- `morganfp.txt`: list of drug morgan fingerprints
- `pro_ctd.txt`: list of protein descriptors

#### `BioKG/` directory
- `data_folds/`: 10 folds training set and test set in the three scenarios
    - `warm_start_1_10/` 
    - `drug_coldstart/` 
    - `protein_coldstart/` 
- `kg.csv`: supporting knowledge graph data
- `dti.csv`: whole DTI dataset
- `comp_struc.csv`: drugbank id and smiles of drugs
- `pro_seq.csv`: sequences of proteins 
- `fp_df.csv`: list of drug morgan fingerprints
- `prodes_df.csv`: list of protein descriptors

#### `hetionet/` directory
- `data_folds/`: 10 folds training set and test set in the three scenarios
    - `warm_start_1_10/` 
    - `drug_coldstart/` 
    - `protein_coldstart/` 
- `kg.csv`: supporting knowledge graph data
- `dti.csv`: whole DTI dataset
- `map_drugs_df`: drugbank id and smiles of drugs
- `pro_seq.csv`: sequences of proteins 
- `fp_df.csv`: list of drug morgan fingerprints
- `prodes_df.csv`: list of protein descriptors

#### `luo's_dataset/` directory
- `data_folds/`: 10 folds training set and test set in the three scenarios
    - `warm_start_1_1/` 
    - `warm_start_1_10/` 
    - `drug_coldstart/` 
    - `protein_coldstart/` 
- `mapping/`: related mappings and similarity matrix (https://github.com/luoyunan/DTINet) 
    - `protein.txt`: list of protein names
    - `disease.txt`: list of disease names
    - `se.txt`: list of side effect names
    - `drug_dict_map`: a complete ID mapping between drug names and DrugBank ID
    - `protein_dict_map`: a complete ID mapping between protein names and UniProt ID
    - `Similarity_Matrix_Drugs.txt` 	: Drug similarity scores based on chemical structures of drugs
    - `Similarity_Matrix_Proteins.txt` 	: Protein similarity scores based on primary sequences of proteins
- `feature/`: related features used in methods
    - `drug_smiles.csv`: drugbank id and smiles
    - `seq.txt`: list of protein sequences
    - `morganfp.txt`: list of drug morgan fingerprints
    - `pro_ctd.txt`: list of protein descriptors

#### `eg_model/` directory
We provided a pre-trained kge model for example.
- `dismult_400_warm_1_10.pkl`


# Prerequisites
#### Operating system: Linux
#### Programing language: python
#### KGE_NFM & NFM dependencies
```
- python 3.6
- pandas '1.1.5'
- numpy '1.18.4'
- scikit-learn '0.24.1'
- tensorflow '1.15.0'
- ampligraph '1.3.2'
- deepctr '0.8.4'
```
#### baseline dependencies
- RF & KGE_RF (included in KGE_NFM&NFM dependencies)
- MPNN_CNN & DeepDTI:
    - source: https://github.com/kexinhuang12345/DeepPurpose 
    ```
    - deeppurpose '0.0.9' 
    - torch '1.6.0+cu101'
    ```
- DTINet: 
    - source: https://github.com/luoyunan/DTINet
    - note: in this work, we run the DTINet in a python environment, which need Linux system and python2. Importantly, this method requires the [Inductive Matrix Completion](http://bigdata.ices.utexas.edu/software/inductive-matrix-completion/) (IMC) library. More detailed information about the installation of this method could be found in the source code of the DTINet.
- DTiGEMS: 
    - source: https://github.com/MahaThafar/DTiGEMSplus
- TriModel: 
    - source: http://drugtargets.insight-centre.org/



# Example (kge_nfm.py)

#### A brief presentation of the results:
- return average loss when training kge model
```
Average Loss:   0.475181:   2%|###3                       | 1/50 [01:10<57:31, 70.44s/epoch]
```
- return performance(mrr) on training set of DTI for early stopping (kge_model in `eg_model/`)
```
In [35]:     roc = roc_auc(test_label,test_score)
    ...:     pr = pr_auc(test_label,test_score)
    ...:     print(roc)
    ...:     print(pr)
0.8731770833333332
0.44079654835037246
```

- nfm training process (`patience=10`)

```
In [45]: roc_nfm,pr_nfm,pred_y = train_nfm(feature_columns,train_model_input,train_label,test_model_input,test_label,patience)
Train on 44851 samples
Epoch 1/2000
44851/44851 - 2s - loss: 0.5332 - precision: 0.0976
Epoch 2/2000
44851/44851 - 1s - loss: 0.4143 - precision: 0.0000e+00
Epoch 3/2000
44851/44851 - 1s - loss: 0.3456 - precision: 0.0000e+00
Epoch 4/2000
44851/44851 - 1s - loss: 0.3443 - precision: 0.0000e+00
Epoch 5/2000
44851/44851 - 1s - loss: 0.3470 - precision: 0.0000e+00
Epoch 6/2000
44851/44851 - 1s - loss: 0.3382 - precision: 0.0000e+00
......
Epoch 279/2000
44851/44851 - 1s - loss: 0.0758 - precision: 0.9248
Epoch 280/2000
44851/44851 - 1s - loss: 0.0753 - precision: 0.9327
Epoch 281/2000
44851/44851 - 1s - loss: 0.0796 - precision: 0.9155
Epoch 282/2000
44851/44851 - 1s - loss: 0.0764 - precision: 0.9276
Epoch 283/2000
44851/44851 - 1s - loss: 0.0739 - precision: 0.9127
```

- reutrn results as type of roc_auc & pr_auc
```
0.9812476679104477
0.8803416284646345
```