import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



list_csv = glob.glob('csv/*.csv')
dfs = []

for i in (list_csv):
    df = pd.read_csv(i)
    dfs.append(df)

column_names = dfs[0].columns.tolist()

print(len(dfs))


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
