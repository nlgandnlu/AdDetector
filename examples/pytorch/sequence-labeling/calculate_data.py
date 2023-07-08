import os
import pandas as pd
from statistics import mean
folder='batch_results/'
names=os.listdir(folder)
names=[n for n in names if '.xlsx' in n]
dfs = [pd.read_excel(folder+x) for x in names]

out_test_index=[3,11,12]
df_tongji=[]
for n,df in zip(names,dfs):
    out_test_up=[]
    out_test_down=[]
    test_mean_up=0
    test_mean_down=0
    for i in range(14):
        if i in out_test_index:
            out_test_up.append(float(df.iloc[i, 4]))
            out_test_down.append(float(df.iloc[i, 8]))
        elif i==13:
            continue
        else:
            test_mean_up+=float(df.iloc[i, 4])/10
            test_mean_down+=float(df.iloc[i, 8])/10
    df_tongji.append([n,test_mean_up,out_test_up[0],out_test_up[1],out_test_up[2],mean(out_test_up),
                        test_mean_down,out_test_down[0],out_test_down[1],out_test_down[2],mean(out_test_down)])
out=pd.DataFrame(df_tongji)
out.to_excel(folder+'tongji.xlsx', index=False)