import os
import shutil
import random
import pandas as pd

src_dir = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork/' 
# train_dir = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/baseline_data_20shot/train/'
# test_dir = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/baseline_data_20shot/test/'
des_dir = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork_nonmixed_100m/' 

imgs_list = os.listdir(src_dir) # 40.922053_-73.900081_513d9be7fdc9f03587007b6e_NewYork.JPG

lu_code_list = list(range(1,7))
# [1, 2, 3, 4, 5, 6]

lu_list = {'residential':[], 
           'commercial':[], 
           'industrial':[], 
           'recreation':[], 
           'transportation':[], 
           'greenfield':[]
           }

lu_code = {'residential':1, 'commercial':2, 'industrial':3, 'recreation':4, 'transportation':5, 'greenfield':6}

# 40.773889_-73.964388_513d7e64fdc9f035870073c8_NewYork.JPG,residential,1.99,2,1,commercial;residential;
df = pd.read_csv('acc_prompts/ensemble_prompt_acc_allpolygons_100m.txt', 
                 sep = ',', 
                 header = None, 
                 thousands = ',', 
                 names=['img','pred_top1_label','diversity','num_osm_labels','bool_correct','osm_labels_list'])

for index, row in df.iterrows():
    if int(row['num_osm_labels'] == 1): 
        shutil.copy(src_dir + row['img'], des_dir + row['img'])

print("done") 

