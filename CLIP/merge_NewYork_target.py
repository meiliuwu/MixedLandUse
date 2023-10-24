import os
import shutil

dest_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork_target/'

# land use
landuse_detected_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/landuse_detected/'

# # buildings
buildings_detected_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/buildings_detected/'

# # natural 
natural_detected_path = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/natural_detected/' 

target_paths = [landuse_detected_path, buildings_detected_path, natural_detected_path]

img_target_set = set()
for path in target_paths:
	# for each path, loop its images
	imgs_list = os.listdir(path)
	for img_name in imgs_list:
		# format: 40.87826_-73.842067_513d9ca0fdc9f03587007de4_NewYork_recreation.JPG
		tokens = img_name.split('_')
		img_prefix = tokens[0] + '_' + tokens[1] + '_' + tokens[2] 
		if img_prefix not in img_target_set:
			img_target_set.add(img_prefix)
			img_rename = img_prefix + '_NewYork.JPG'
			shutil.copy(path + img_name, dest_path + img_rename)
