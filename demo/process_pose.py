import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config
from new_utils import get_pose_sparse_img, load_frame_count

from split_train_test_video import *

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

model = get_model('vgg19')
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

splitter = UCF101_splitter(path='/home/molly/two-stream-action-recognition/UCF_list/',split='04')
train_video, test_video = splitter.split_video()

frame_count = load_frame_count()

for video_name in train_video.keys():
    print(video_name)
    print(frame_count[video_name])
    for index in range(frame_count[video_name]):
        get_pose_sparse_img(video_name, index+1, model)

for video_name in test_video.keys():
    print(video_name)
    print(frame_count[video_name])
    for index in range(frame_count[video_name]):
        get_pose_sparse_img(video_name, index+1, model)


'''
#test_image = './readme/ski.jpg'

oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image

with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

print("paf shape =" , paf.shape)
print("heatmap shape = ", heatmap.shape)
print("Ori_img shape = ", oriImg.shape)
print("im_scale = ", im_scale)

humans = paf_to_pose_cpp(heatmap, paf, cfg)
print("humans shape = ", len(humans))
#print(humans[0].body_parts.keys())
#print(humans[0].body_parts[0])

out = draw_humans(oriImg, humans)
#cv2.imwrite('result.png',out)
cv2.imwrite('/home/molly/UCF_data/pose_flow/result_test.png',out)
'''

'''
"./readme/ski.jpg"
paf shape = (46, 49, 38)
heatmap shape =  (46, 49, 19)
Ori_img shape =  (674, 712, 3)
im_scale =  0.5459940652818991
humans shape =  5
dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17])
BodyPart:0-(0.09, 0.40) score=0.97

for jpeg2_256 imgs:
paf shape = (46, 62, 38)        #38 pairs of nodes (limbs)4
heatmap shape =  (46, 62, 19)   #0-17 keys denoting body_parts and 18 for background
Ori_img shape =  (256, 342, 3)
im_scale =  1.4375

'''
