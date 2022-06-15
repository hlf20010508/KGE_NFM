import os
workspace=os.getcwd()
os.system('wget -P %s https://github.com/hlf20010508/KGE_NFM/releases/download/data/yamanishi_08.zip'%workspace)
os.system('cd %s && unzip yamanishi_08.zip && rm yamanishi_08.zip'%workspace)