import torchvision
import numpy as np
import os

import random
from PIL import Image, ImageDraw
from collections import Counter
import matplotlib as plt
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict



# draw_bbox_and_save(target["id_corresponding"],target["boxes"],visual_direc,img_folder)
# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack(images, 0)
#     targets = [{key: torch.as_tensor(value) for key, value in t.items()} for t in targets]
#     return images, targets
def collate_fn(batch):
    return tuple(zip(*batch))
class CustomDataset(Dataset):
    def __init__(self, img_folder, data, transforms=None):
        self.img_folder = img_folder
        self.data=data
        # If no transforms are provided, include a default transform that converts the image to tensor
        self.transforms = transforms if transforms is not None else T.Compose([T.ToTensor()])

    def __getitem__(self, idx):
        img_id = self.data[idx][0]
        target=self.data[idx][1]

        img_name = '0' * (12 - len(str(img_id))) + str(img_id) 
        img_path = os.path.join(self.img_folder, img_name + '.jpg')  # Assuming JPEG images
        img = Image.open(img_path).convert("RGB")

        # Apply transforms to the image
        if self.transforms:
            img = self.transforms(img)

        # Ensure the target is in the correct format (tensor)
        # target['boxes'] = torch.stack(target['boxes'], dim=0).to(dtype=torch.float32)
        # print("Chiii",type(target['boxes']))
        # print(type(target["boxes"][0]))

        temp = torch.empty(0, 4)

        for x in target['boxes']:
            temp = torch.vstack((temp, x))
        
        target['boxes'] = temp
        # target['boxes'] = torch.tensor(torch.tensor(x) for x in target['boxes']
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.data)

class TrainVal:
    def __init__(self, state):
        self.state = state

    def train(self, args, target, device, img_folder):
        id_images = target["id_corresponding"]
        bboxs, labels = target["boxes"], target["labels"]
        data_set={}
        for i in range(len(id_images)):
            id=int(id_images[i].item())
            if id not in data_set :
                data_set[id]={'boxes': [bboxs[i]],'labels':[labels[i]]}
            else:
                data_set[id]['boxes'].append(bboxs[i])
                data_set[id]['labels'].append(labels[i])
        unique_id = list(data_set.keys())
        
        train_ids, val_ids = train_test_split(unique_id, test_size=args.test_size,train_size=1-args.test_size)


        train_data = [(img_id, data_set[img_id]) for img_id in train_ids]
        val_data = [(img_id, data_set[img_id]) for img_id in val_ids]

        transforms= T.Compose([T.Resize((448, 448)),T.ToTensor()])
        #customize the dataset
        cus_train_data= CustomDataset(img_folder,train_data,transforms=transforms)
        cus_val_data=CustomDataset(img_folder,val_data,transforms=transforms)
        
        # DataLoader
        train_loader = DataLoader(cus_train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(cus_val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = args.num_classes + 1  # +1 for background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        model.to(device)

        # Training loop
        for epoch in range(args.num_epochs):
            model.train()
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {losses.item()}')


# class train_val():
#     def __init__(self,state):
#         self.state=state
#     def train(self,args,target,device,img_folder):
#         id_image=target["id_corresponding"]
#         targets= devide_data(target["boxes"],target["labels"])
        
#         train_ids,val_inds =train_test_split()

#         model= torchvision.models.detection.fasterrcnn_resnet50_fpn( pretrained = True)
#         num_classes=args.num_classes
#         in_feature=model.roi_heads.box_predictor_cls_score.infeatures
#         model.roi_heads.box_predictor=FastRCNNPredictor(in_feature,num_classes)

#         optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
#         num_epochs=args.num_epochs
#         batch_size=args.batch_size

#         model.to(device)
#         for epochs in range(num_epochs):
#             epochs =0
#             for i in range(len)