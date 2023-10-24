import os
import clip
import torch
from PIL import Image
import numpy as np
#from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from torchvision.datasets import CIFAR100
from tqdm import tqdm
#from sklearn import preprocessing
import itertools

import time

# get the start time
start = time.time()

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

class CustomImageDataset(Dataset):
	def __init__(self, img_dir, transform=None, target_transform=None):
		self.img_dir = img_dir
		self.imgs_list = os.listdir(img_dir)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.imgs_list)

	def __getitem__(self, idx):
		filename = self.imgs_list[idx] # e.g., 1.356434_103.750897_50f56202fdc9f065f0005859_1.JPG
		image = Image.open(os.path.join(self.img_dir, filename)).convert("RGB")
		#label = int(filename.split('_')[3].split('.')[0]);
		#label = int(filename.split('_')[0]);

		if self.transform:
			image = self.transform(image)
		# if self.target_transform:
		# 	label = self.target_transform(label)
		return image #, label

# "__ land use"
# "__ use"
# "__ area"
# "__ place"
# "for _"
# prompt ensembling

log_file = open('log_file_zeroshot_ViT-L14_cityname.txt','w+')
res_file = open('res_file_zeroshot_ViT-L14_cityname.txt','w+')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

#model, preprocess = clip.load('ViT-B/32', device)
#model, preprocess = clip.load('ViT-B/16', device)
#model, preprocess = clip.load('RN50x64', device) 
model, preprocess = clip.load('ViT-L/14', device)
log_file.write("running on %s\n"% device)

# Load the dataset
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/samples/" #training and testing should be separated
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/images_citycodes/" #training and testing should be separated
train_path = "/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork_nonmixed_100m/"
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/paintings_citycode/"
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/im2gps3k_56cities_openview/"
#test_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/test/"   #training and testing should be separated

train_list = os.listdir(train_path)
#test_list = os.listdir(test_path)

log_file.write("training %d images\n"% len(train_list))
#log_file.write("testing %d images\n"% len(test_list))

train = CustomImageDataset(train_path, transform=preprocess)
#test = CustomImageDataset(test_path, transform=preprocess)

lu_list = ['residential', 'commercial', 'industrial', 'recreation', 'greenfield', 'transportation']
log_file.write("%d land use labels in total \n"% len(lu_list))

def get_features(dataset, lu_list):
	all_features = []
	all_labels = []

	with torch.no_grad():
		# use: 0.048
		# purpose: 0.320
		# no: 0.632

		# use
		# text_tokens1 = clip.tokenize([f"{c} use" for c in lu_list]).cuda()
		# text_features1 = model.encode_text(text_tokens1).float()
		# # purpose
		# text_tokens2 = clip.tokenize([f"{c} purpose" for c in lu_list]).cuda()
		# text_features2 = model.encode_text(text_tokens2).float()
		# # no template
		# text_tokens3 = clip.tokenize([f"{c}" for c in lu_list]).cuda()
		# text_features3 = model.encode_text(text_tokens3).float()

		# text_features = text_features1 * 0.048 + text_features2 * 0.320 + text_features3 * 0.632

		# spatial awareness prompt engineering
		text_tokens = clip.tokenize([f"{c} in New York" for c in lu_list]).cuda()
		text_features = model.encode_text(text_tokens).float()

		#for images, labels in tqdm(DataLoader(dataset, batch_size=110, num_workers=4)): #3083 very slow!!
		for images in tqdm(DataLoader(dataset, batch_size=64, num_workers=4)): # 36: in total there will be 3083 bacthes, but out of memory!!

			image_input = torch.tensor(np.stack(images)).cuda()
			image_features = model.encode_image(image_input).float()
			all_features.append(image_features)
			#all_labels.append(labels) # label convert into [0,1,2,3,4,...]
	
	return torch.cat(all_features), text_features #, torch.cat(all_labels).cpu().numpy()

# Calculate the image features
image_features, text_features = get_features(train, lu_list)
#test_features, test_labels = get_features(test)

# Pick the top 6 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

Print the result of each image
for i, pred in enumerate(similarity):
	filename = train.imgs_list[i]
    
	values, indices = pred.topk(6)

	#write res_log.txt
	res_file.write("%s"% (filename))

	for value, index in zip(values, indices):
		lu_pred = lu_list[index]
		lu_pred_perc = value
		res_file.write(",%s,%.2f"% (lu_pred, lu_pred_perc))
	res_file.write("\n")

log_file.close()
res_file.close()


# if (device == "cuda"): torch.cuda.synchronize()    # <---------------- extra line
# end = time.time()
# print("Run time [s] without GPU: ", end - start)