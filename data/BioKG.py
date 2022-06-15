import os
workspace=os.getcwd()
os.system('wget -P %s https://github.com/hlf20010508/KGE_NFM/releases/download/data/BioKG.zip'%workspace)
os.system('cd %s && unzip BioKG.zip && rm BioKG.zip'%workspace)