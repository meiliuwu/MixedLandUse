import pandas as pd
import math
import os
import shutil

df = pd.read_csv("prompt_ensemble_acc_eachlayer_foundone_clean.txt", 
                 sep = ',', 
                 header = None, 
                 thousands = ',', 
                 names=['img','pred_label','diversity','num_groundtrue_labels','bool_acc'])


df_nonOSM = df[df['bool_acc'] == -1] # imgs having no OSM land use data

imgs_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork/'
dest_path = "/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork_nonOSM/"

for index, row in df_nonOSM.iterrows():
	shutil.copy(imgs_path + row['img'], dest_path + row['img'])
