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
from sklearn import preprocessing
import itertools

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
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
		filename = self.imgs_list[idx] # e.g., 40.593963_-73.95257_513d7c1efdc9f03587006db6_NewYork_residential.JPG
		image = Image.open(os.path.join(self.img_dir, filename)).convert("RGB")
		label = filename.split('_')[-1].split('.')[0]; # e.g., residential
		#label = int(filename.split('_')[0]);

		if self.transform:
			image = self.transform(image)
		# if self.target_transform:
		# 	label = self.target_transform(label)
		return image, label

# "__ land use"
# "__ use"
# "__ area"
# "__ place"
# "for _"
# prompt ensembling

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

#model, preprocess = clip.load('ViT-B/32', device)
#model, preprocess = clip.load('ViT-B/16', device)
#model, preprocess = clip.load('RN50x64', device) 
model, preprocess = clip.load('ViT-L/14', device)

# Load the dataset
train_path = "/home/ai4sg/meiliu/CLIP/CLIP/landuse/all_detected/"

train_list = os.listdir(train_path)
#test_list = os.listdir(test_path)

train = CustomImageDataset(train_path, transform=preprocess)
#test = CustomImageDataset(test_path, transform=preprocess)

lu_list = ['residential', 'commercial', 'industrial', 'recreation', 'greenfield', 'transportation']
lu_code = {'residential':1, 'commercial':2, 'industrial':3, 'recreation':4, 'greenfield':5, 'transportation':6}
code_lu = {1:'residential', 2:'commercial', 3:'industrial', 4:'recreation', 5:'greenfield', 6:'transportation'}

def get_features(dataset):
	all_features = []
	all_labels = []
	with torch.no_grad():
		for images, labels in tqdm(DataLoader(dataset, batch_size=110, num_workers=4)):
			
			image_input = torch.tensor(np.stack(images)).cuda()
			image_features = model.encode_image(image_input) #.float()

			all_features.append(image_features)
			all_labels.extend(labels)
			#labels2codes = [int(lu_code[label]) for label in labels]
			#all_labels.append(labels2codes) # label convert into [0,1,2,3,4,...]

	# le = preprocessing.LabelEncoder()
	# flat_list = list(itertools.chain(*all_labels))
	# targets = le.fit_transform(flat_list)
	# # targets: array([0, 1, 2, 3])
	# targets = torch.as_tensor(targets)
	# # targets: tensor([0, 1, 2, 3])
			
	return torch.cat(all_features).cpu().numpy(), all_labels

#test_features, test_labels = get_features(train)
test_features, lu_labels = get_features(train)

# lu_labels = []
# for code in test_labels: 
# 	lu_labels.append(code_lu[code])

test_features_embedded = TSNE(n_components=2, 
								verbose=1, 
								perplexity=500, 
								n_iter=10000, 
								learning_rate='auto', 
								init='pca'
							 ).fit_transform(test_features)

df_subset = pd.DataFrame()
df_subset['TSNE Dimension 1'] = test_features_embedded[:,0]
df_subset['TSNE Dimension 2'] = test_features_embedded[:,1]
df_subset['Land Use'] = lu_labels

plt.figure(figsize=(10,8))
sns.scatterplot(x="TSNE Dimension 1", y="TSNE Dimension 2",
						hue="Land Use",
						palette=sns.color_palette("hls", 6),
						data=df_subset,
						legend="full",
						alpha=0.5
					  )
plt.legend(#bbox_to_anchor=(1.02, 1), 
	loc='upper right', 
	borderaxespad=0, 
	ncol=1)
#plt.show()
plt.tight_layout()
plt.savefig("t-SNE_perp500_inter10k.png", dpi=150, bbox_inches='tight')

print("done")


# perp500_inter10k









# perp500_inter5k







# perp500_inter4k
# [t-SNE] Computing 1501 nearest neighbors...
# [t-SNE] Indexed 1740 samples in 0.001s...
# [t-SNE] Computed neighbors for 1740 samples in 0.301s...
# [t-SNE] Computed conditional probabilities for sample 1000 / 1740
# [t-SNE] Computed conditional probabilities for sample 1740 / 1740
# [t-SNE] Mean sigma: 3.531054
# /home/ai4sg/anaconda3/envs/clip/lib/python3.7/site-packages/sklearn/manifold/_t_sne.py:986: FutureWarning: The PCA initialization in TSNE will change to have the standard deviation of PC1 equal to 1e-4 in 1.2. This will ensure better convergence.
#   FutureWarning,
# [t-SNE] KL divergence after 250 iterations with early exaggeration: 42.952076
# [t-SNE] KL divergence after 1300 iterations: 0.432967






# perp500_inter3k







# perp500_inter2k








# perp500_inter1k
