import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from PyBioMed import Pyprotein
from PyBioMed.PyProtein import CTD


#the drug structure is extracted from drugbank
################################################################ FP
drug = pd.read_csv('./data/yamanishi_08/791drug_struc.csv')

smiles = drug['smiles'].values
mols = [Chem.MolFromSmiles(x) for x in smiles]
fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024,) for x in mols]
fp_array = np.array(fp)
np.savetxt('morganfp.txt',fp_array,delimiter=',')


#the protein sequence is extracted from uniprot&kegg database
################################################################ CTD
pro = pd.read_csv('./data/yamanishi_08/989proseq.csv')
seq = pro['seq'].values

protein_descriptor=[]
for i in range(len(seq)):
    ctd = CTD.CalculateCTD(seq[i]).values()
    protein_descriptor.append(list(ctd))

protein_descriptor = np.array(protein_descriptor)
np.savetxt('pro_ctd.txt',protein_descriptor,delimiter=',')