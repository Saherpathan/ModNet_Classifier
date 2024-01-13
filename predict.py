import argparse
import json

import numpy as np
import torch
import torchvision
from torch import nn,optim
from torchvision import models,transforms
from PIL import Image


def check_load(filepath,gpu):
    checkpoint = torch.load(filepath,map_location='cuda' if args.gpu=='gpu' else 'cpu')
    if checkpoint['arch']=='vgg19':
       model=models.vgg19(pretrained=True)

    elif checkpoint['arch'] =='vgg16':
        model=models.vgg16(pretrained=True)

    elif checkpoint['arch']=='alexnet':
        model=models.alexnet(pretrained=True)

    else:
        raise ValueError ("Please select: vgg19/vgg16/alexnet")

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_idx'])
    model.class_to_idx = checkpoint['class_to_idx']

    for para in model.parameters():
        para.requires_grad = False

    return model

def process_image(image):
    processes=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(size=224),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    image=processes(image)
    return image

def predict(image_path,model,topk=5):
    model = model.cuda()
    model.eval()
    image = Image.open(image_path)
    images = process_image(image)
    images = images.unsqueeze(0)
    images = images.cuda() if args.gpu=='gpu' else images.cpu()

    with torch.no_grad():
        output = model.forward(images)

    prob = torch.exp(output)
    top_prob,top_indices = prob.topk(topk,dim=1)
    idx_to_class = {idx: class_name for class_name, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices[0].cpu().numpy()]
    return top_prob,top_classes

def load_file(filename):
    with open(filename,'r') as f:
        category_name = json.load(f,strict=False)
    return category_name



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction part of COMMAND LINE APPLICATION")
    parser.add_argument('--category_names', default='cat_to_name.json',type=str)
    parser.add_argument('--checkpoint',default='checkpt.pth',action='store')
    parser.add_argument('--topk',default=5,type=int)
    parser.add_argument('--gpu',type=str)
    parser.add_argument('--filepath',default='C:\\Users\Saher\PycharmProjects\\Udacity_ND\\flowers\\test\\76\image_02484.jpg',type=str)


    args = parser.parse_args()
    model = check_load(args.checkpoint,args.gpu)
    cat_to_name = load_file(args.category_names)
    no_of_class = args.topk
    image_path = args.filepath

    if args.gpu=='gpu':
        model.cuda()
    else:
        model.cpu()

    probs,classes = predict(image_path,model)
    name = [cat_to_name[index] for index in classes]
    names = np.arange(len(name))
    probab = np.array(probs.cpu()).flatten()

    for i in range(no_of_class):
        print(f'Predicted class is {names[i]} ',
              f'Having probability of {probab[i]*100} %')
