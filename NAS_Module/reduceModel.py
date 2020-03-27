from torchvision import models
import torch
import sys


print(dir(models))
vgg = models.vgg11(pretrained=True)
print(vgg)


from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from PIL import Image

img = Image.open("/dataset/ImageNet/val/n01531178/ILSVRC2012_val_00032359.JPEG")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)


for param in vgg.parameters():
    param.requires_grad = False
layers = list(vgg.features.children())[:-1]



#for k in range(len(layers)):
#    print(k,layers[k])


for name, param in vgg.named_parameters():
    print (name,param.requires_grad,param.data.shape,param.data.min())

import copy
features_3_weight = copy.deepcopy(vgg.state_dict()["features.3.weight"][:])
features_3_bias = copy.deepcopy(vgg.state_dict()["features.3.bias"][:])
features_6_weight = copy.deepcopy(vgg.state_dict()["features.6.weight"][:])
features_6_bias = copy.deepcopy(vgg.state_dict()["features.6.bias"][:])


layers[3] = torch.nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
layers[6] = torch.nn.Conv2d(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

features = torch.nn.Sequential(*layers).cuda()

vgg.features = features

print("="*100)
for name, param in vgg.named_parameters():
    print (name,param.requires_grad,param.data.shape,param.data.min())

print("="*100)
print(vgg.state_dict()["features.6.weight"][:].shape)
print(features_6_weight.shape)

vgg.state_dict()["features.3.weight"][:] = features_3_weight[0:96,:,:,:]
vgg.state_dict()["features.3.bias"][:] = features_3_bias[0:96]
vgg.state_dict()["features.6.weight"][:] = features_6_weight[:,0:96,:,:]
vgg.state_dict()["features.6.bias"][:] = features_6_bias[:]

print("="*100)
for name, param in vgg.named_parameters():
    print (name,param.requires_grad,param.data.shape,param.data.min())




'''alexnet.eval()
out = alexnet(batch_t)
print(out.shape)
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())


_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
'''
