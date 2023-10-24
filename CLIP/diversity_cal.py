import pandas as pd
import math

df = pd.read_csv("res_file_zeroshot_ViT-L14_purpose-Brooklyn.txt", 
                 sep = ',', 
                 header = None, 
                 thousands = ',', 
                 names=['img','l1','p1','l2','p2','l3','p3','l4','p4','l5','p5','l6','p6'])

for index, row in df.iterrows():
    summing = 0.0

    if row['p1'] != 0.0 : 
        ele = row['p1']
        summing += (-1.0) * ele * math.log(ele)

    if row['p2'] != 0.0 : 
        ele = row['p2']
        summing += (-1.0) * ele * math.log(ele)

    if row['p3'] != 0.0 : 
        ele = row['p3']
        summing += (-1.0) * ele * math.log(ele)

    if row['p4'] != 0.0 : 
        ele = row['p4']
        summing += (-1.0) * ele * math.log(ele)

    if row['p5'] != 0.0 : 
        ele = row['p5']
        summing += (-1.0) * ele * math.log(ele)

    if row['p6'] != 0.0 : 
        ele = row['p6']
        summing += (-1.0) * ele * math.log(ele)

    df.at[index, 'diversity'] = math.exp(summing)

df.to_csv('res_file_zeroshot_ViT-L14_purpose-Brooklyn-diversity_50m.csv')

