import os
import shutil
import random
import pandas as pd

src_dir = '/home/bpeng/meiliu/landuse/NewYork/' 
Bronx_dir = '/home/bpeng/meiliu/landuse/Bronx/'
Brooklyn_dir = '/home/bpeng/meiliu/landuse/Brooklyn/'
Manhattan_dir = '/home/bpeng/meiliu/landuse/Manhattan/'
Queens_dir = '/home/bpeng/meiliu/landuse/Queens/'

imgs_list = os.listdir(src_dir) # 40.922053_-73.900081_513d9be7fdc9f03587007b6e_NewYork.JPG

Bronx_df = pd.read_csv('Bronx_pts.csv', 
                 sep = ',', 
                 header = 0, 
                 thousands = ',', 
                 names=['OID','lat','lon','id','pred_label','diversity','num_groundtruth_labels','bool_acc']) #OID_	lat	lon	id	pred_label	diversity	num_groundtruth_labels	bool_acc

Brooklyn_df = pd.read_csv('Brooklyn_pts.csv', 
                 sep = ',', 
                 header = 0, 
                 thousands = ',', 
                 names=['OID','lat','lon','id','pred_label','diversity','num_groundtruth_labels','bool_acc']) #OID_	lat	lon	id	pred_label	diversity	num_groundtruth_labels	bool_acc
                 
Manhattan_df = pd.read_csv('Manhattan_pts.csv', 
                 sep = ',', 
                 header = 0, 
                 thousands = ',', 
                 names=['OID','lat','lon','id','pred_label','diversity','num_groundtruth_labels','bool_acc']) #OID_	lat	lon	id	pred_label	diversity	num_groundtruth_labels	bool_acc
                 
Queens_df = pd.read_csv('Queens_pts.csv', 
                 sep = ',', 
                 header = 0, 
                 thousands = ',', 
                 names=['OID','lat','lon','id','pred_label','diversity','num_groundtruth_labels','bool_acc']) #OID_	lat	lon	id	pred_label	diversity	num_groundtruth_labels	bool_acc

count = 0
for img in imgs_list:
    if (img.split('_')[2]) in list(Bronx_df['id']):
        shutil.copy(src_dir + img, Bronx_dir + img)
        count = count + 1
    elif (img.split('_')[2]) in list(Brooklyn_df['id']):
        shutil.copy(src_dir + img, Brooklyn_dir + img)
        count = count + 1
    elif (img.split('_')[2]) in list(Manhattan_df['id']):
        shutil.copy(src_dir + img, Manhattan_dir + img)
        count = count + 1
    elif (img.split('_')[2]) in list(Queens_df['id']):
        shutil.copy(src_dir + img, Queens_dir + img)
        count = count + 1

print(count) 


