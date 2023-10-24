import geopandas as gpd
from shapely.geometry import Point
import os
import shutil
import glob

res_file = open('img_groundtrue_eachlayer_foundone.txt','w+')

imgs_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork/'
imgs_list = os.listdir(imgs_path)

# land use
landuse_detected_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/landuse_detected/'

# buildings
buildings_detected_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/buildings_detected/'

# natural 
natural_detected_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/natural_detected/' 

for img_name in imgs_list:

	tokens = img_name.split('_')
	# format: 40.753654_-73.848825_513d7ebbfdc9f0358700747a_NewYork.JPG
	img_prefix = tokens[0] + '_' + tokens[1] + '_' + tokens[2]
	lu_set = set()

	# 1. find if it is in landuse folder
	img_tokens = landuse_detected_path + img_prefix + '_*.JPG'
	for file in glob.glob(img_tokens):
		# format: 40.710147_-73.908517_513d7ba1fdc9f03587006c17_NewYork_residential.JPG 
		lu_label = file.split('_')[-1].split('.')[0]
		lu_set.add(lu_label)

	# 2. find if it is in buildings folder
	img_tokens = buildings_detected_path + img_prefix + '_*.JPG'
	for file in glob.glob(img_tokens):
		# format: 40.710147_-73.908517_513d7ba1fdc9f03587006c17_NewYork_residential.JPG 
		lu_label = file.split('_')[-1].split('.')[0]
		lu_set.add(lu_label)

	# 3. find if it is in natrual folder
	img_tokens = natural_detected_path + img_prefix + '_*.JPG'
	for file in glob.glob(img_tokens):
		# format: 40.710147_-73.908517_513d7ba1fdc9f03587006c17_NewYork_residential.JPG 
		lu_label = file.split('_')[-1].split('.')[0]
		lu_set.add(lu_label)

	#write res_log.txt
	res_file.write("%s,%d"% (img_name, len(lu_set)))

	for value in lu_set:
		res_file.write(",%s"% (value))
	res_file.write("\n")

res_file.close()