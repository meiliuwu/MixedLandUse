import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#from collections import OrderedDict
import torch

# city_code = {'New York': 0, 'Singapore': 1, 'London': 2, 'Prague': 3, 'Melbourne': 4, 'Belo Horizonte': 5, 'Barcelona': 6, 'Sydney': 7, 'Montreal': 8, 'Rio de Janeiro': 9, 'Munich': 10, 'Milan': 11, 'Tel Aviv': 12, 'Dublin': 13, 'Zagreb': 14, 'Rome': 15, 'Sao Paulo': 16, 'Bratislava': 17, 'Seattle': 18, 'Chicago': 19, 'Stockholm': 20, 'Kyoto': 21, 'Johannesburg': 22, 'Boston': 23, 'Moscow': 24, 'Minneapolis': 25, 'Toronto': 26, 'Kiev': 27, 'Copenhagen': 28, 'Valparaiso': 29, 'Los Angeles': 30, 'Atlanta': 31, 'Madrid': 32, 'Houston': 33, 'Tokyo': 34, 'Cape Town': 35, 'Amsterdam': 36, 'Philadelphia': 37, 'Gaborone': 38, 'Paris': 39, 'Portland': 40, 'Mexico City': 41, 'Warsaw': 42, 'Washington D.C.': 43, 'Santiago': 44, 'Guadalajara': 45, 'San Francisco': 46, 'Helsinki': 47, 'Berlin': 48, 'Taipei': 49, 'Bangkok': 50, 'Bucharest': 51, 'Denver': 52, 'Hong Kong': 53, 'Glasgow': 54, 'Lisbon': 55}
# 'Singapore': 1, 'Rio de Janeiro': 9, 'Mexico City': 41, 'Guadalajara': 45
# 'New York': 0
# 'Chicago': 19
# 'London': 2
# 'Sydney': 7
# 'Seattle': 18
# 'Toronto': 26
# 'Los Angeles': 30
# 'Houston': 33

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Loading the model
# clip.available_models() will list the names of available CLIP models.
import clip
clip.available_models()
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']

print("Available image encoder models:", clip.available_models())

#model, preprocess = clip.load("ViT-B/32")
#model, preprocess = clip.load("RN50x16")
model, preprocess = clip.load("ViT-L/14")
#model, preprocess = clip.load("RN50x64")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# images in skimage to use and their textual descriptions
# descriptions = {
#     "page": "a page of text about segmentation",
#     "chelsea": "a facial photo of a tabby cat",
#     "astronaut": "a portrait of an astronaut with the American flag",
#     "rocket": "a rocket standing on a launchpad",
#     "motorcycle_right": "a red motorcycle standing in a garage",
#     "camera": "a person looking at a camera on a tripod",
#     "horse": "a black-and-white silhouette of a horse", 
#     "coffee": "a cup of coffee on a saucer"
# }


# Get the list of all files and directories
# in the root directory
#path = "data/placeplus2/" 
path = "/home/ai4sg/meiliu/CLIP/CLIP/data/placeplus2/train_4_shot_baseline/33/"
dir_list = os.listdir(path)
  
print("Files and directories in '", path, "' :") 
  
# print the list
print(len(dir_list))

original_images = []
images = []
texts = ["residential", "commercial", "industrial", "transportation", "recreational", "agricultural", "water"]

# plt.figure(figsize=(16, 5))

for filename in dir_list:
    image = Image.open(os.path.join(path, filename)).convert("RGB")
    #print(type(image))

    original_images.append(image)
    images.append(preprocess(image))
    #texts.append(descriptions[name])


# Building features
# We normalize the images, tokenize each text input, and run the forward pass of the model to get the image and text features.
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize([f"{desc} land use" for desc in texts]).cuda()

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

# normalize the features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

# save the results
plt.figure(figsize=(18, 64))

for i, image in enumerate(original_images):
    plt.subplot(25, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(25, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [texts[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

#plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(wspace=0.4, hspace=0.3, left=0.1, bottom=0.1, right=0.96, top=0.96)
#plt.show()
plt.savefig('results_streetview_ViT-L14_city33_Houston.png', dpi=150)
print("done")