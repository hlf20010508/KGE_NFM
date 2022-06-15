import os
workspace=os.getcwd()
os.system('wget -P %s https://github.com/hlf20010508/KGE_NFM/releases/download/data/luo.s_dataset.zip'%workspace)
os.system('cd %s && unzip luo.s_dataset.zip && rm luo.s_dataset.zip'%workspace)