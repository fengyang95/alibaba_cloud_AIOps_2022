import pandas as pd
df=pd.read_csv('./preliminary_venus_dataset.csv')
print(df['module'].values)
from collections import Counter
counter=Counter(df['module'].values)
print(f"counter:{counter}")