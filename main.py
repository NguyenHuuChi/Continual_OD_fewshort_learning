from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from Visualization.visual_box import *
from Datasets.coco_conti import build
import argparse
from model.Faster_RCNN import TrainVal

parser=argparse.ArgumentParser(description='')

parser.add_argument('--dataset', default='coco_base', type=str, help='')
parser.add_argument('--coco_path',default="/vinserver_user/22chi.nh/newprojecthuy/data",help="data path")
parser.add_argument("--masks", default=False,help="")
parser.add_argument("--cache_mode",default=False,help="")
parser.add_argument("--status", default="train",help="")
parser.add_argument("--num_classes",default=90 , help="")
parser.add_argument("--num_epochs", default=5, help="")
parser.add_argument("--batch_size",default=32, help='')
parser.add_argument("--test_size",default=0.25,type=float)

args = parser.parse_args()

visual_dir ="/vinserver_user/22chi.nh/Continual_learning"
if args.dataset == "coco_base":
    visual_direc = f"{visual_dir}/Visulation_image/COCO"     
else:
    visual_direc = f"{visual_dir}/Visulation_image/BBD"   
       
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset , img_folder, ann_file=build("val",args)
img,target,path_oringin_image=dataset[:]

train=TrainVal(args.status)
train.train(args,target=target, device=device,img_folder=img_folder)
