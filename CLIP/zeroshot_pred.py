import os
import clip
import torch
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from torchvision.datasets import CIFAR100
from tqdm import tqdm
from sklearn import preprocessing
import itertools

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

city_code = {'New York': 0, 'Singapore': 1, 'London': 2, 'Prague': 3, 'Melbourne': 4, 'Belo Horizonte': 5, 'Barcelona': 6, 'Sydney': 7, 'Montreal': 8, 'Rio de Janeiro': 9, 'Munich': 10, 'Milan': 11, 'Tel Aviv': 12, 'Dublin': 13, 'Zagreb': 14, 'Rome': 15, 'Sao Paulo': 16, 'Bratislava': 17, 'Seattle': 18, 'Chicago': 19, 'Stockholm': 20, 'Kyoto': 21, 'Johannesburg': 22, 'Boston': 23, 'Moscow': 24, 'Minneapolis': 25, 'Toronto': 26, 'Kiev': 27, 'Copenhagen': 28, 'Valparaiso': 29, 'Los Angeles': 30, 'Atlanta': 31, 'Madrid': 32, 'Houston': 33, 'Tokyo': 34, 'Cape Town': 35, 'Amsterdam': 36, 'Philadelphia': 37, 'Gaborone': 38, 'Paris': 39, 'Portland': 40, 'Mexico City': 41, 'Warsaw': 42, 'Washington D.C.': 43, 'Santiago': 44, 'Guadalajara': 45, 'San Francisco': 46, 'Helsinki': 47, 'Berlin': 48, 'Taipei': 49, 'Bangkok': 50, 'Bucharest': 51, 'Denver': 52, 'Hong Kong': 53, 'Glasgow': 54, 'Lisbon': 55}

code_city = {0: 'New York', 1: 'Singapore', 2: 'London', 3: 'Prague', 4: 'Melbourne', 5: 'Belo Horizonte', 6: 'Barcelona', 7: 'Sydney', 8: 'Montreal', 9: 'Rio de Janeiro', 10: 'Munich', 11: 'Milan', 12: 'Tel Aviv', 13: 'Dublin', 14: 'Zagreb', 15: 'Rome', 16: 'Sao Paulo', 17: 'Bratislava', 18: 'Seattle', 19: 'Chicago', 20: 'Stockholm', 21: 'Kyoto', 22: 'Johannesburg', 23: 'Boston', 24: 'Moscow', 25: 'Minneapolis', 26: 'Toronto', 27: 'Kiev', 28: 'Copenhagen', 29: 'Valparaiso', 30: 'Los Angeles', 31: 'Atlanta', 32: 'Madrid', 33: 'Houston', 34: 'Tokyo', 35: 'Cape Town', 36: 'Amsterdam', 37: 'Philadelphia', 38: 'Gaborone', 39: 'Paris', 40: 'Portland', 41: 'Mexico City', 42: 'Warsaw', 43: 'Washington D.C.', 44: 'Santiago', 45: 'Guadalajara', 46: 'San Francisco', 47: 'Helsinki', 48: 'Berlin', 49: 'Taipei', 50: 'Bangkok', 51: 'Bucharest', 52: 'Denver', 53: 'Hong Kong', 54: 'Glasgow', 55: 'Lisbon'}

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
		label = int(filename.split('_')[3].split('.')[0]);
		#label = int(filename.split('_')[0]);

		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label

log_file = open('log_file_zeroshot_ViT-L14_prompt_ensemble_all.txt','w+')
res_file = open('res_file_zeroshot_ViT-L14_prompt_ensemble_all.txt','w+')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load('ViT-B/32', device)
#model, preprocess = clip.load('ViT-B/16', device)
#model, preprocess = clip.load('RN50x64', device) 
model, preprocess = clip.load('ViT-L/14', device)
log_file.write("running on %s\n"% device)

# Load the dataset
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/samples/" #training and testing should be separated
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/images_citycodes/" #training and testing should be separated
train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/test/"
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/paintings_citycode/"
#train_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/im2gps3k_56cities_openview/"
#test_path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/test/"   #training and testing should be separated

train_list = os.listdir(train_path)
#test_list = os.listdir(test_path)

log_file.write("training %d images\n"% len(train_list))
#log_file.write("testing %d images\n"% len(test_list))

train = CustomImageDataset(train_path, transform=preprocess)
#test = CustomImageDataset(test_path, transform=preprocess)

city_list = ['New York', 'Singapore', 'London', 'Prague', 'Melbourne', 'Belo Horizonte', 'Barcelona', 'Sydney', 'Montreal', 'Rio de Janeiro', 'Munich', 'Milan', 'Tel Aviv', 'Dublin', 'Zagreb', 'Rome', 'Sao Paulo', 'Bratislava', 'Seattle', 'Chicago', 'Stockholm', 'Kyoto', 'Johannesburg', 'Boston', 'Moscow', 'Minneapolis', 'Toronto', 'Kiev', 'Copenhagen', 'Valparaiso', 'Los Angeles', 'Atlanta', 'Madrid', 'Houston', 'Tokyo', 'Cape Town', 'Amsterdam', 'Philadelphia', 'Gaborone', 'Paris', 'Portland', 'Mexico City', 'Warsaw', 'Washington D.C.', 'Santiago', 'Guadalajara', 'San Francisco', 'Helsinki', 'Berlin', 'Taipei', 'Bangkok', 'Bucharest', 'Denver', 'Hong Kong', 'Glasgow', 'Lisbon']
#city_list = ['Warsaw', 'Rome', 'Paris', 'Moscow', 'London', 'Lisbon', 'Dublin', 'Copenhagen', 'Berlin', 'Barcelona', 'Amsterdam']
log_file.write("%d cities\n"% len(city_list))

def get_features(dataset, city_list):
	all_features = []
	all_labels = []

	with torch.no_grad():

		text_tokens1 = clip.tokenize([f"at {c}" for c in city_list]).cuda()
		text_tokens2 = clip.tokenize([f"from {c}" for c in city_list]).cuda()
		text_tokens3 = clip.tokenize([f"in {c}" for c in city_list]).cuda()

		text_features1 = model.encode_text(text_tokens1) #.float()
		text_features2 = model.encode_text(text_tokens2) #.float()
		text_features3 = model.encode_text(text_tokens3) #.float()

		#print("at from in")
		#print(softmax(scores)) #[0.13471716 0.28519622 0.58008662]
		#text_features = text_features1 * 0.135 + text_features2 * 0.285 + text_features3 * 0.580

		#print("to city_of none at from in")
		# ['0.000', '0.099', '0.102', '0.108', '0.228', '0.463']
		text_tokens4 = clip.tokenize([f"city of {c}" for c in city_list]).cuda()
		text_tokens5 = clip.tokenize([f"{c}" for c in city_list]).cuda()

		text_features4 = model.encode_text(text_tokens4) #.float()
		text_features5 = model.encode_text(text_tokens5) #.float()

		text_features = text_features1 * 0.108 + text_features2 * 0.228 + text_features3 * 0.463 + text_features4 * 0.099 + text_features5 * 0.102

		#for images, labels in tqdm(DataLoader(dataset, batch_size=110, num_workers=4)): #3083 very slow!!
		for images, labels in tqdm(DataLoader(dataset, batch_size=110, num_workers=4)): # 36: in total there will be 3083 bacthes, but out of memory!!

			image_input = torch.tensor(np.stack(images)).cuda()
			image_features = model.encode_image(image_input) #.float() 
			all_features.append(image_features)
			all_labels.append(labels) # label convert into [0,1,2,3,4,...]
	
	return torch.cat(all_features), text_features, torch.cat(all_labels).cpu().numpy()

# Calculate the image features
image_features, text_features, true_labels = get_features(train, city_list)
#test_features, test_labels = get_features(test)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

predictions = []
top5_acc_cnt = 0
# Print the result of each image
for i, pred in enumerate(similarity):
	filename = train.imgs_list[i]
	city_true = code_city[int(filename.split('_')[3].split('.')[0])]
	#city_true = code_city[int(filename.split('_')[0])]
	city_true2 = code_city[true_labels[i]]

	values, indices = pred.topk(5)
	top1_index_in_citylist = indices[0]
	city_pred = city_list[top1_index_in_citylist]

	predictions.append(city_code[city_pred])

	#write res_log.txt
	res_file.write("%s,%s,%s,%s"% (filename, city_true, city_true2, city_pred))

	found = False
	for value, index in zip(values, indices):
		city_pred = city_list[index]
		city_pred_perc = value
		res_file.write(",%s,%.2f"% (city_pred, city_pred_perc))
	    #print(f"{city_list[index]:>16s}: {100 * value.item():.2f}%")

		if found == False and city_pred == city_true:
			found = True

	res_file.write("\n")
	if found == True:
		top5_acc_cnt = top5_acc_cnt + 1

# all_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

# Evaluate zero-shot
arr = np.array(np.array(predictions) == np.array(true_labels))
top1_acc = 1.0 * np.count_nonzero(arr) / len(arr) * 100
log_file.write("Top1 Accuracy = %.3f\n"% top1_acc)
#log_file.write("C=0.316\n")

top5_acc = 1.0 * top5_acc_cnt / len(arr) * 100
log_file.write("Top5 Accuracy = %.3f\n"% top5_acc)

log_file.close()
res_file.close()



# import os
# import clip
# import torch
# from PIL import Image
# import skimage
# import IPython.display
# import matplotlib.pyplot as plt
# import numpy as np

# #import Double
# #from torchvision.datasets import CIFAR100

# log_file = open('log_file.txt','w+')
# res_file = open('res_file.txt','w+')

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# log_file.write("running on %s\n"% device)
# model, preprocess = clip.load('ViT-L/14', device)

# model.cuda().eval()
# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size

# # log_file.write("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# # log_file.write("Input resolution:", input_resolution)
# # log_file.write("Context length:", context_length)
# # log_file.write("Vocab size:", vocab_size)

# city_list = ['NewYork', 'Singapore', 'London', 'Prague', 'Melbourne', 'BeloHorizonte', 'Barcelona', 'Sydney', 'Montreal', 'RioDeJaneiro', 'Munich', 'Milan', 'TelAviv', 'Dublin', 'Zagreb', 'Rome', 'SaoPaulo', 'Bratislava', 'Seattle', 'Chicago', 'Stockholm', 'Kyoto', 'Johannesburg', 'Boston', 'Moscow', 'Minneapolis', 'Toronto', 'Kiev', 'Copenhagen', 'Valparaiso', 'LosAngeles', 'Atlanta', 'Madrid', 'Houston', 'Tokyo', 'CapeTown', 'Amsterdam', 'Philadelphia', 'Gaborone', 'Paris', 'Portland', 'MexicoCity', 'Warsaw', 'WashingtonDC', 'Santiago', 'Guadalajara', 'SanFrancisco', 'Helsinki', 'Berlin', 'Taipei', 'Bangkok', 'Bucharest', 'Denver', 'HongKong', 'Glasgow', 'Lisbon']
# log_file.write("%d cities\n"% len(city_list))

# # Get the list of all files and directories
# #path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/test/"
# path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/images/"
# dir_list = os.listdir(path)
# log_file.write("%d images\n"% len(dir_list))

# log_file.close()

# images = []
# img_filenames = []
# true_labels = []
# pred_lables = []

# for filename in dir_list:
# 	image = Image.open(os.path.join(path, filename)).convert("RGB")
# 	#print(type(image))
# 	images.append(preprocess(image))

# 	tokens = filename.split('_')
# 	#lat = Double.parseDouble(tokens[0])
# 	#lon = Double.parseDouble(tokens[1])
# 	#imgId = tokens[2]
# 	img_filenames.append(filename)

# 	city = tokens[3].split('.')[0];
# 	true_labels.append(city)

# print("starting zero-shot running...")
# # Building features
# # We normalize the images, tokenize each text input, and run the forward pass of the model to get the image and text features.
# image_input = torch.tensor(np.stack(images)).cuda()
# #text_tokens = clip.tokenize([f"in {c}" for c in city_list]).cuda()
# text_tokens = torch.cat([clip.tokenize(f"in {c}") for c in city_list]).cuda()

# with torch.no_grad():
# 	image_features = model.encode_image(image_input).float()
# 	text_features = model.encode_text(text_tokens).float()

# # normalize the features
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)

# text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

# print("end of zero-shot running...")

# print("starting to write the results of zero-shot predictions...")
# top5_acc_cnt = 0
# top1_acc_cnt = 0
# for i, img_top_labels in enumerate(top_labels):
# 	pred_label = city_list[img_top_labels[0].numpy()]
# 	pred_lables.append(pred_label)
# 	res_file.write("\n#%d image\n"% i)
# 	res_file.write("image filename: %s\n"% img_filenames[i])
# 	res_file.write("true label: %s\n"% true_labels[i])
# 	res_file.write("predicted label: %s\n"% pred_label)

# 	if true_labels[i] == pred_label:
# 		top1_acc_cnt = top1_acc_cnt + 1
# 	top5_labels = []
# 	for topk in range(5):
# 		topk_label = city_list[img_top_labels[topk].numpy()]
# 		topk_label_prob = top_probs[i][topk].numpy()
# 		top5_labels.append(topk_label)
# 		res_file.write("top %d label: %s\t prob: %.3f\n" % (topk + 1, topk_label, topk_label_prob))
# 		#res_file.write("top " + (topk + 1) + " label: " + topk_label + " prob: " + topk_label_prob)
# 	if true_labels[i] in top5_labels: 
# 		top5_acc_cnt = top5_acc_cnt + 1

# # Evaluate zero-shot
# #accuracy = np.mean((true_labels == pred_lables).astype(np.float)) * 100.
# #res_file.write(f"Accuracy = {accuracy:.3f}")
# top1_acc = 1.0 * top1_acc_cnt / len(dir_list) * 100
# res_file.write("\nTop1 Accuracy = %.3f\n"% top1_acc)

# top5_acc = 1.0 * top5_acc_cnt / len(dir_list) * 100
# res_file.write("Top5 accuracy = %.3f\n"% top5_acc)
# res_file.close()

# print("end of writing the results of zero-shot predictions...")

# Print the result
# print("\nimage id:", imgId)
# print("\nTop predictions:\n")
# for value, index in zip(top_probs, top_labels):
# 	print(f"{city_list[index]:>16s}: {100 * value.item():.2f}%")


#text_inputs = torch.cat([clip.tokenize(f"in {c}") for c in city_list]).to(device)
# text_features = model.encode_text(text_inputs)

# test_labels = []
# pred_lables = []


	# # an filename example: "55.778234_37.647113_513e1d72fdc9f035870099de_Moscow.JPG"
	# tokens = filename.path.split('_')
	# #lat = Double.parseDouble(tokens[0])
	# #lon = Double.parseDouble(tokens[1])
	# imgId = tokens[2]
	# city = tokens[3].split('.')[0];

	# test_labels.append(city)

	# # Prepare the inputs
	# image_input = preprocess(file).unsqueeze(0).to(device)

	# # Calculate features
	# with torch.no_grad():
	# 	image_features = model.encode_image(image_input)
	    
	# # Pick the top 5 most similar labels for the image
	# image_features /= image_features.norm(dim=-1, keepdim=True)
	# text_features /= text_features.norm(dim=-1, keepdim=True)
	# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
	# values, indices = similarity[0].topk(5)

	# pred_lables.append(city_list[indices[0]])

	# # Print the result
	# print("\nimage id:", file.path)
	# print("\nTop predictions:\n")
	# for value, index in zip(values, indices):
	# 	print(f"{city_list[index]:>16s}: {100 * value.item():.2f}%")

