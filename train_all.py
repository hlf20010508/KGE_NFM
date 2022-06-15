import os
import time
workspace=os.getcwd()
time_str = time.strftime('%Y%m%d%H%M')
os.system('cd %s && pipenv run python deepdti.py | tee ./logs/deepdti/%s.log'%(workspace,time_str))
os.system('cd %s && pipenv run python kge_nfm.py | tee ./logs/kge_nfm/%s.log'%(workspace,time_str))
os.system('cd %s && pipenv run python kge_rf.py | tee ./logs/kge_rf/%s.log'%(workspace,time_str))