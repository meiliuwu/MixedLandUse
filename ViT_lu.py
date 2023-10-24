import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms

# from torchvision.models import (
#     vit_b_16, ViT_B_16_Weights,
#     # vit_h_14, ViT_H_14_Weights,
# )

import time
start = time.time()

log_file = open('log_file_vit_baseline_5shot.txt','w+')
#log_file3k = open('log_file_linearprob_baseline_im2gps3k.txt','w+')
#res_file = open('res_file_linearprob_baseline_1shot.txt','w+')

# check if CUDA is available
train_on_gpu = torch.cuda.is_available() # False
# train_on_gpu = False

if not train_on_gpu:
    log_file.write('CUDA is not available.  Training on CPU ...\n')
else:
    log_file.write('CUDA is available!  Training on GPU ...\n')


# Load and Transform our Data
# define training and test data directories
data_dir = '/home/bpeng/meiliu/landuse/baseline_data_5shot/'
#train_dir = os.path.join(data_dir, 'train_1_shot_baseline/')
#train_dir = os.path.join(data_dir, 'train_2_shot_baseline/')
#train_dir = os.path.join(data_dir, 'train_4_shot_baseline/')
#train_dir = os.path.join(data_dir, 'train_8_shot_baseline/')
#train_dir = os.path.join(data_dir, 'train_16_shot_baseline/')
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

#test_dir = os.path.join(data_dir, 'test_baseline/')
#test_dir_im2gps = os.path.join(data_dir, 'test_baseline_im2gps/')
#test_dir_im2gps3k = os.path.join(data_dir, 'test_baseline_im2gps3k/')

# classes are folders in each directory with these names
l = [i for i in range(1,7)]
classes = [str(x) for x in l]

print(classes)

log_file.write('Num classes: %d\n'% len(classes))
# load and transform data using ImageFolder

# VGG-16 Takes 224x224 images as input, so we resize all of them
# data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
#                                      transforms.ToTensor()])

data_transform = transforms.Compose([transforms.Resize(224), 
                                     transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
log_file.write('Num training images: %d\n'% len(train_data))
log_file.write('Num test images: %d\n'% len(test_data))


# DataLoaders
# define dataloader parameters
batch_size = 20
num_workers = 16

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)


# Load the pretrained model from pytorch
# model_variant, model_weights_to_use = vit_b_16, ViT_B_16_Weights   # NO IMAGE SIZE MISMATCH ERROR

# Final Classifier Layer
import torch.nn as nn

model = models.vit_b_16(weights='DEFAULT')

# setup for six class classification
num_ftrs = model.heads[-1].in_features
model.heads[-1] = nn.Linear(num_ftrs, len(classes))


device = torch.device("cuda:0" if train_on_gpu else "cpu")
print(f"using device: {device}")
model = model.to(device)



# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
# last_layer = nn.Linear(n_inputs, len(classes))

# resnet152.fc = last_layer

# # if GPU is available, move the model to GPU
# if train_on_gpu:
#     resnet152.cuda()

# check to see that your last layer produces the expected number of outputs
#print(resnet152.fc.out_features)


import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
# optimizer = optim.SGD(resnet152.fc.parameters(), lr=0.001)
optimizer = optim.SGD(model.heads[-1].parameters(), lr=0.001)

# # Training
# number of epochs to train the model
n_epochs = 32

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # model by default is set to train
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        
        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            log_file.write('Epoch %d, Batch %d loss: %.16f\n' %
                           (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0


# Testing
# track test loss 
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

model.eval() # eval mode

# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class

    #for i in range(batch_size + 1):
    for i in range(target.data.size(dim=0)): 
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
log_file.write('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        log_file.write('Test Accuracy of %5s: %.2f%% (%2d/%2d)\n' % (
                       classes[i], 100. * class_correct[i] / class_total[i],
                       np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        log_file.write('Test Accuracy of %5s: N/A (no training examples)\n' % (classes[i]))

log_file.write('\nTest Accuracy (Overall): %.2f%% (%2d/%2d)\n' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

print("done")
# insert some code to do something here

# if train_on_gpu: torch.cuda.synchronize()    # <---------------- extra line
# end = time.time()
# print("Run time [s] without GPU 5shot: ",end-start)

















# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (6): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (7): Bottleneck(
#       (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (3): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (4): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (5): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (6): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (7): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (8): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (9): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (10): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (11): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (12): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (13): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (14): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (15): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (16): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (17): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (18): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (19): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (20): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (21): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (22): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (23): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (24): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (25): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (26): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (27): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (28): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (29): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (30): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (31): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (32): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (33): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (34): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (35): Bottleneck(
#       (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): Bottleneck(
#       (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (downsample): Sequential(
#         (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#     (2): Bottleneck(
#       (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=2048, out_features=1000, bias=True)
# )
