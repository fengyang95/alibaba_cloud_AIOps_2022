import pandas as pd
import numpy as np

if __name__ == '__main__':
    dfs = []
    res = []
    for path in ['./submit_lgb_5_tuna_00.csv',
                 './submit_lgb_5_tuna_01.csv',
                 './submit_lgb_5_tuna_02.csv',
                 './submit_lgb_5_tuna_03.csv',
                 './submit_lgb_5_tuna_04.csv',
                 './submit_lgb_5_tuna_05.csv',
                 './submit_lgb_5_tuna_06.csv',
                 './submit_lgb_5_tuna_07.csv',
                 './submit_lgb_5_tuna_08.csv',
                 #'./submit_lgb_5_tuna_09.csv',
                 ]:
        df = pd.read_csv(path)
        print(len(df))
        res.append(df['label'].values)
    result = np.array(res)
    labels=[]
    c = 0

    for r in result.T:
        # print(f"r:{r}")
        l=np.argmax(np.bincount(r))
        labels.append(l)
        if len(np.unique(r))!=1:
            c+=1
        print(f"r:{r} label:{l}")
    print(f"c:{c}")
    df=pd.read_csv('./submit_lgb_5_tuna_00.csv')
    df['label']=np.array(labels).astype(int)
    df.to_csv('submit_merge.csv',index=False)


