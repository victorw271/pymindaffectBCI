import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

list_csv = glob.glob('csv/**/*.csv')

dfs = []

path_list = []

for path in list_csv:
    path_list.append(path.split(os.sep))

time = [path[1] for path in path_list] 

for i in (list_csv):
    df = pd.read_csv(i)
    dfs.append(df)

column_names = dfs[0].columns.tolist()
column_names2 = dfs[4].columns.tolist()

list_csv2 = glob.glob('csv/**/metadata.csv')

mds = []

for i in (list_csv2):
    md = pd.read_csv(i, index_col=False)
    mds.append(md)

column_names3 = mds[0].columns.tolist()

k_audc = []
l_audc = []
p_audc = []

s = dfs[0]['AUDC']
g = dfs[1]['AUDC']
f = dfs[3]['AUDC']

sha = []
cm_id = []
branch = []

for count, md in enumerate(mds):
    sha.append(md['sha'])
    branch.append(md['branch'])
    cm_id.append(count)

print(mds[0]['branch'])
#print(sha)
#print(cm_id)
#print(branch)

#print(column_names3)
#print(dfs[0]['AUDC'])
#print(mds[0]['current_date_and_time'])
#print(column_names)
#print(column_names2)
#print(path_list)
#print(len(time))
#print(time)
#print(path_list)
#print(dfs[4].columns.tolist())


#file = column_names[0]
#clsfr = column_names[1]
#audc = column_names[2]
#ave_audc = column_names[3]
#bsl_audc = column_names[4]

#kaggle = dfs[0]
#lowlands = dfs[1]
#plos_one = dfs[2]

#k_audc = kaggle['AUDC']
#l_audc = lowlands['AUDC']
#p_audc = plos_one['AUDC']

#print(dfs[0]['AUDC'])

#source = pd.DataFrame([[20.500, 15.513, 50.667], [57.833, 23.077, 60.667],[46.000, 55.436, 70.333],[36.667, 60.256, 80.222],[55.000, 85.718, 90.456]],
                          #columns=['Kaggle', 'Plos_one', 'Lowlands'], index=pd.RangeIndex(5, name='Commit-ID'))

#print(source)

#source2 = source.reset_index().melt('Commit-ID', var_name='dataset', value_name=audc)

#print(source2)

#for i in range(len(dfs)):
    #print(dfs[i])

#print(dfs[0]["AUDC"])

#print(np.random.randn(20, 3))
